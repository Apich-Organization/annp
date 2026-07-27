#include "common.cuh"
#include <cfloat>
#include <cstdint>

/**
 * Industrial-Grade Parallel Particle Q-Router Kernel.
 * 1. Coalesced Global Memory Access & Alignment Safety.
 * 2. Dynamic Warp Reduction Sizing & Zero Garbage Memory Invariants.
 * 3. Race-Free Block-Level Reductions with Explicit Barrier Synchronization.
 * 4. Stream-Ordered Device Buffer Management (cudaMallocAsync / cudaFreeAsync).
 */

extern "C" __global__ void particle_router_kernel(
    const float* __restrict__ p_in,          // [batch_size, d_head]
    const float* __restrict__ p_out,         // [batch_size, d_head]
    const float* __restrict__ routing_table, // Transposed layout [d_head, num_neighbors]
    const float* __restrict__ gumbel_noise,  // [batch_size, num_neighbors]
    int* __restrict__ chosen_neighbor,       // [batch_size]
    bool* __restrict__ halting_flags,        // [batch_size]
    int batch_size,
    int d_head,
    int num_neighbors,
    float temperature,
    float epsilon_p,
    float epsilon_h,
    uint16_t min_hop,
    const ParticleCudaHeader* __restrict__ headers
) {
    int pid = blockIdx.x; // 1 Block per particle batch item
    if (pid >= batch_size) return;

    int tid = threadIdx.x;
    int active_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    int active_warps_pad = (active_warps + 3) & ~3;
    int num_neighbors_pad = (num_neighbors + 3) & ~3;

    const float* pin = p_in + pid * d_head;
    const float* pout = p_out + pid * d_head;
    const float* gnoise = gumbel_noise + pid * num_neighbors;

    // Fully Dynamic 16-byte Aligned Shared Memory Partitioning
    extern __shared__ __align__(16) float s_mem_raw[];
    float* s_reduce_tmp = s_mem_raw;
    float* s_logits     = s_reduce_tmp + active_warps_pad;
    float* s_probs      = s_logits + num_neighbors_pad;
    float* s_gmax       = s_probs + num_neighbors_pad;
    int* s_best_k       = reinterpret_cast<int*>(s_gmax + active_warps_pad);

    for (int i = tid; i < active_warps_pad; i += blockDim.x) {
        s_reduce_tmp[i] = 0.0f;
        s_gmax[i] = -FLT_MAX;
        s_best_k[i] = 0;
    }
    for (int i = tid; i < num_neighbors_pad; i += blockDim.x) {
        s_logits[i] = -FLT_MAX;
        s_probs[i] = 0.0f;
    }
    __syncthreads();

    // 1. Cooperative Delta p calculation: ||p_out - p_in||_2
    float diff_sq = 0.0f;
    bool aligned = (((uintptr_t)pin | (uintptr_t)pout) & 15) == 0 && (d_head % 4 == 0);

    if (aligned) {
        int d_head4 = d_head / 4;
        const float4* pin4 = reinterpret_cast<const float4*>(pin);
        const float4* pout4 = reinterpret_cast<const float4*>(pout);

        for (int i = tid; i < d_head4; i += blockDim.x) {
            float4 in_v = __ldg(pin4 + i);
            float4 out_v = __ldg(pout4 + i);
            float dx = out_v.x - in_v.x;
            float dy = out_v.y - in_v.y;
            float dz = out_v.z - in_v.z;
            float dw = out_v.w - in_v.w;
            diff_sq += dx * dx + dy * dy + dz * dz + dw * dw;
        }
    } else {
        for (int i = tid; i < d_head; i += blockDim.x) {
            float diff = __ldg(pout + i) - __ldg(pin + i);
            diff_sq += diff * diff;
        }
    }

    float w_sum = warp_reduce_sum(diff_sq);
    if ((tid & 31) == 0) {
        s_reduce_tmp[tid >> 5] = w_sum;
    }
    __syncthreads();

    float delta_p_sq = 0.0f;
    for (int w = 0; w < active_warps; ++w) {
        delta_p_sq += s_reduce_tmp[w];
    }
    float delta_p = sqrtf(delta_p_sq);
    __syncthreads();

    // 2. Coalesced Logits calculation
    for (int k = tid; k < num_neighbors; k += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            dot += __ldg(pout + d) * __ldg(routing_table + d * num_neighbors + k);
        }
        s_logits[k] = dot;
    }
    __syncthreads();

    // 3. Block-Level Parallel Max Reduction across active warps
    float local_max = -FLT_MAX;
    for (int k = tid; k < num_neighbors; k += blockDim.x) {
        local_max = fmaxf(local_max, s_logits[k]);
    }
    float w_max = warp_reduce_max(local_max);

    __syncthreads();
    if ((tid & 31) == 0) {
        s_reduce_tmp[tid >> 5] = w_max;
    }
    __syncthreads();

    float global_max = s_reduce_tmp[0];
    for (int w = 1; w < active_warps; ++w) {
        global_max = fmaxf(global_max, s_reduce_tmp[w]);
    }

    // 4. Parallel Softmax Exp & Sum Reduction across active warps
    float inv_temp = 1.0f / fmaxf(temperature, 1e-4f);
    float local_exp_sum = 0.0f;
    for (int k = tid; k < num_neighbors; k += blockDim.x) {
        float exp_val = __expf((s_logits[k] - global_max) * inv_temp);
        s_probs[k] = exp_val;
        local_exp_sum += exp_val;
    }

    float w_exp_sum = warp_reduce_sum(local_exp_sum);
    __syncthreads();
    if ((tid & 31) == 0) {
        s_reduce_tmp[tid >> 5] = w_exp_sum;
    }
    __syncthreads();

    float global_exp_sum = 0.0f;
    for (int w = 0; w < active_warps; ++w) {
        global_exp_sum += s_reduce_tmp[w];
    }
    float inv_sum = 1.0f / (global_exp_sum + 1e-8f);
    float noise_base = (num_neighbors > 0) ? (0.05f / (float)num_neighbors) : 0.0f;

    // 5. Parallel Entropy & Gumbel-Softmax evaluation across active warps
    float local_entropy = 0.0f;
    float local_gumbel_max = -FLT_MAX;
    int local_best_k = 0;

    for (int k = tid; k < num_neighbors; k += blockDim.x) {
        float raw_prob = s_probs[k] * inv_sum;
        float prob_val = 0.95f * raw_prob + noise_base;
        s_probs[k] = prob_val;
        if (prob_val > 1e-10f) {
            local_entropy -= prob_val * log2f(prob_val + 1e-10f);
        }

        float g_val = logf(prob_val + 1e-10f) + gnoise[k];
        if (g_val > local_gumbel_max) {
            local_gumbel_max = g_val;
            local_best_k = k;
        }
    }

    float w_entropy = warp_reduce_sum(local_entropy);

    // Correct Warp-Level Parallel Argmax Reduction with Active Thread Masking
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(FULL_WARP_MASK, local_gumbel_max, offset);
        int other_k = __shfl_down_sync(FULL_WARP_MASK, local_best_k, offset);
        if (other_max > local_gumbel_max) {
            local_gumbel_max = other_max;
            local_best_k = other_k;
        }
    }

    __syncthreads();
    if ((tid & 31) == 0) {
        s_reduce_tmp[tid >> 5] = w_entropy;
        s_gmax[tid >> 5] = local_gumbel_max;
        s_best_k[tid >> 5] = local_best_k;
    }
    __syncthreads();

    float global_entropy = 0.0f;
    for (int w = 0; w < active_warps; ++w) {
        global_entropy += s_reduce_tmp[w];
    }

    // Thread 0 collects warp results and writes final decision
    if (tid == 0) {
        float best_g_max = s_gmax[0];
        int best_k = s_best_k[0];
        for (int w = 1; w < active_warps; ++w) {
            float w_gmax = s_gmax[w];
            if (w_gmax > best_g_max) {
                best_g_max = w_gmax;
                best_k = s_best_k[w];
            }
        }
        chosen_neighbor[pid] = best_k;

        ParticleCudaHeader header = headers[pid];
        bool is_halted = (header.halted != 0);
        if (!is_halted && header.hop_count >= min_hop) {
            if (delta_p < epsilon_p && global_entropy < epsilon_h) {
                is_halted = true;
            }
        }
        halting_flags[pid] = is_halted;
    }
}

struct RouterDeviceBufferPool {
    float* d_p_in = nullptr;
    float* d_p_out = nullptr;
    float* d_routing_table = nullptr;
    float* d_gnoise = nullptr;
    int* d_chosen = nullptr;
    bool* d_halting = nullptr;
    ParticleCudaHeader* d_headers = nullptr;
    size_t p_cap = 0;
    size_t gnoise_cap = 0;
    size_t batch_cap = 0;
    size_t rt_cap = 0;
    cudaStream_t last_stream = nullptr;

    ~RouterDeviceBufferPool() {
        if (d_p_in) { cudaFree(d_p_in); d_p_in = nullptr; }
        if (d_p_out) { cudaFree(d_p_out); d_p_out = nullptr; }
        if (d_gnoise) { cudaFree(d_gnoise); d_gnoise = nullptr; }
        if (d_chosen) { cudaFree(d_chosen); d_chosen = nullptr; }
        if (d_halting) { cudaFree(d_halting); d_halting = nullptr; }
        if (d_headers) { cudaFree(d_headers); d_headers = nullptr; }
        if (d_routing_table) { cudaFree(d_routing_table); d_routing_table = nullptr; }
        cudaGetLastError();
    }

    void ensure_capacity(size_t batch_size, size_t d_head, size_t num_neighbors, cudaStream_t stream) {
        if (last_stream != stream && last_stream != nullptr) {
            cudaStreamSynchronize(last_stream);
        }
        last_stream = stream;

        size_t p_needed = batch_size * d_head;
        size_t gnoise_needed = batch_size * num_neighbors;

        if (p_needed > p_cap || batch_size > batch_cap || gnoise_needed > gnoise_cap) {
            if (d_p_in) cudaFreeAsync(d_p_in, stream);
            if (d_p_out) cudaFreeAsync(d_p_out, stream);
            if (d_gnoise) cudaFreeAsync(d_gnoise, stream);
            if (d_chosen) cudaFreeAsync(d_chosen, stream);
            if (d_halting) cudaFreeAsync(d_halting, stream);
            if (d_headers) cudaFreeAsync(d_headers, stream);

            batch_cap = batch_size * 2 + 512;
            p_cap = p_needed * 2 + 1024;
            gnoise_cap = gnoise_needed * 2 + 1024;

            CHECK_CUDA(safe_cuda_malloc_async(&d_p_in, p_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_p_out, p_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_gnoise, gnoise_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_chosen, batch_cap * sizeof(int), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_halting, batch_cap * sizeof(bool), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_headers, batch_cap * sizeof(ParticleCudaHeader), stream));
        }
        size_t rt_size = d_head * num_neighbors;
        if (rt_size > rt_cap) {
            if (d_routing_table) cudaFreeAsync(d_routing_table, stream);
            rt_cap = rt_size * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_routing_table, rt_cap * sizeof(float), stream));
        }
    }
};

static thread_local RouterDeviceBufferPool g_router_pool;

// C-FFI Wrapper Function with Zero-Copy Device Memory Detection
extern "C" void launch_particle_router(
    const float* p_in,
    const float* p_out,
    const float* routing_table,
    const float* gumbel_noise,
    int* chosen_neighbor,
    bool* halting_flags,
    int batch_size,
    int d_head,
    int num_neighbors,
    float temperature,
    float epsilon_p,
    float epsilon_h,
    uint16_t min_hop,
    const ParticleCudaHeader* headers,
    cudaStream_t stream
) {
    if (batch_size <= 0 || d_head <= 0 || num_neighbors <= 0) return;

    g_router_pool.ensure_capacity(batch_size, d_head, num_neighbors, stream);

    bool was_host_copied = false;
    const float* dev_pin = get_device_ptr(p_in, g_router_pool.d_p_in, batch_size * d_head, stream, &was_host_copied);
    const float* dev_pout = get_device_ptr(p_out, g_router_pool.d_p_out, batch_size * d_head, stream, &was_host_copied);
    const float* dev_rt = get_device_ptr(routing_table, g_router_pool.d_routing_table, d_head * num_neighbors, stream, &was_host_copied);
    const float* dev_gnoise = get_device_ptr(gumbel_noise, g_router_pool.d_gnoise, batch_size * num_neighbors, stream, &was_host_copied);
    const ParticleCudaHeader* dev_hdr = get_device_ptr(headers, g_router_pool.d_headers, batch_size, stream, &was_host_copied);

    int* dev_chosen = get_device_ptr_mut(chosen_neighbor, g_router_pool.d_chosen, batch_size, stream, &was_host_copied);
    bool* dev_halting = get_device_ptr_mut(halting_flags, g_router_pool.d_halting, batch_size, stream, &was_host_copied);

    int threads_per_block = (num_neighbors <= 32) ? 32 : 128;
    int blocks = batch_size; // 1 block per particle
    int num_neighbors_pad = (num_neighbors + 3) & ~3;

    int active_warps = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    int active_warps_pad = (active_warps + 3) & ~3;

    size_t float_count = 3 * active_warps_pad + 2 * num_neighbors_pad;
    size_t shared_mem_bytes = float_count * sizeof(float);

    particle_router_kernel<<<blocks, threads_per_block, shared_mem_bytes, stream>>>(
        dev_pin, dev_pout, dev_rt, dev_gnoise, dev_chosen, dev_halting,
        batch_size, d_head, num_neighbors, temperature, epsilon_p, epsilon_h, min_hop, dev_hdr
    );

    copy_back_if_host(chosen_neighbor, dev_chosen, batch_size, stream);
    copy_back_if_host(halting_flags, dev_halting, batch_size, stream);
}
