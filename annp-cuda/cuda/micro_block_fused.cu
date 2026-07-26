#include "common.cuh"
#include <cuda_runtime.h>

// Block-wide reduction using warp shuffle + shared memory across warps
__device__ inline float block_reduce_sum(float val, float* s_reduce) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    // 1. Intra-warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }

    // Write warp sums to shared memory
    if (lane == 0) {
        s_reduce[wid] = val;
    }
    __syncthreads();

    // 2. Reduce warp sums using the first warp
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (tid < num_warps) ? s_reduce[tid] : 0.0f;

    if (wid == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
        }
    }

    // Broadcast final result from thread 0 to all threads in block
    if (tid == 0) {
        s_reduce[0] = val;
    }
    __syncthreads();

    return s_reduce[0];
}

extern "C" __global__ void fused_micro_block_kernel(
    const float* __restrict__ p_in,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    float* __restrict__ p_out,
    int batch_size,
    int d_head,
    int ffn_dim,
    int kv_len,
    int norm_strategy,
    float alpha,
    float sphere_radius
) {
    int pid = blockIdx.x;
    if (pid >= batch_size) return;

    int tid = threadIdx.x;
    const float* curr_p = p_in + pid * d_head;
    float* curr_out = p_out + pid * d_head;

    int dh_pad = (d_head + 3) & ~3;
    int ffn_pad = (ffn_dim + 3) & ~3;

    // Dynamic Shared Memory Allocation with 16-byte Alignment
    extern __shared__ __align__(16) float s_mem[];
    float* s_p_in       = s_mem;
    float* s_attn       = s_mem + dh_pad;
    float* s_mid        = s_mem + 2 * dh_pad;
    float* s_ffn_inter  = s_mem + 3 * dh_pad;
    float* s_reduce     = s_mem + 3 * dh_pad + ffn_pad; // Buffer for block reductions

    bool aligned = (((uintptr_t)curr_p | (uintptr_t)curr_out | (uintptr_t)k_cache | (uintptr_t)v_cache) & 15) == 0 && (d_head % 4 == 0);

    // Step 0: Load p_in into Shared Memory & initialize s_attn
    if (aligned) {
        int vec_d_head = d_head / 4;
        const float4* curr_p_vec = reinterpret_cast<const float4*>(curr_p);
        float4* s_p_in_vec = reinterpret_cast<float4*>(s_p_in);
        float4* s_attn_vec = reinterpret_cast<float4*>(s_attn);

        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            s_p_in_vec[i] = __ldg(curr_p_vec + i);
            s_attn_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    } else {
        for (int i = tid; i < d_head; i += blockDim.x) {
            s_p_in[i] = __ldg(curr_p + i);
            s_attn[i] = 0.0f;
        }
    }
    __syncthreads();

    // Step 1: Online FlashAttention
    if (kv_len > 0) {
        float m_prev = -1e9f;
        float d_prev = 0.0f;

        for (int k = 0; k < kv_len; ++k) {
            const float* k_ptr = k_cache + k * d_head;
            float val = 0.0f;

            if (aligned) {
                int vec_d_head = d_head / 4;
                const float4* k_ptr_vec = reinterpret_cast<const float4*>(k_ptr);
                float4* s_p_in_vec = reinterpret_cast<float4*>(s_p_in);
                for (int i = tid; i < vec_d_head; i += blockDim.x) {
                    float4 p_v = s_p_in_vec[i];
                    float4 k_v = __ldg(k_ptr_vec + i);
                    val += p_v.x * k_v.x + p_v.y * k_v.y + p_v.z * k_v.z + p_v.w * k_v.w;
                }
            } else {
                for (int i = tid; i < d_head; i += blockDim.x) {
                    val += s_p_in[i] * __ldg(k_ptr + i);
                }
            }

            float dot = block_reduce_sum(val, s_reduce);
            float score = dot * rsqrtf((float)d_head);

            float m_curr = fmaxf(m_prev, score);
            float alpha_scale = __expf(m_prev - m_curr);
            float p_val = __expf(score - m_curr);

            d_prev = d_prev * alpha_scale + p_val;
            m_prev = m_curr;

            const float* v_ptr = v_cache + k * d_head;
            if (aligned) {
                int vec_d_head = d_head / 4;
                const float4* v_ptr_vec = reinterpret_cast<const float4*>(v_ptr);
                float4* s_attn_vec = reinterpret_cast<float4*>(s_attn);
                for (int i = tid; i < vec_d_head; i += blockDim.x) {
                    float4 a_v = s_attn_vec[i];
                    float4 v_v = __ldg(v_ptr_vec + i);
                    s_attn_vec[i] = make_float4(
                        a_v.x * alpha_scale + p_val * v_v.x,
                        a_v.y * alpha_scale + p_val * v_v.y,
                        a_v.z * alpha_scale + p_val * v_v.z,
                        a_v.w * alpha_scale + p_val * v_v.w
                    );
                }
            } else {
                for (int i = tid; i < d_head; i += blockDim.x) {
                    s_attn[i] = s_attn[i] * alpha_scale + p_val * __ldg(v_ptr + i);
                }
            }
        }

        if (d_prev > 1e-8f) {
            float inv_d = 1.0f / d_prev;
            if (aligned) {
                int vec_d_head = d_head / 4;
                float4* s_attn_vec = reinterpret_cast<float4*>(s_attn);
                for (int i = tid; i < vec_d_head; i += blockDim.x) {
                    float4 a_v = s_attn_vec[i];
                    s_attn_vec[i] = make_float4(a_v.x * inv_d, a_v.y * inv_d, a_v.z * inv_d, a_v.w * inv_d);
                }
            } else {
                for (int i = tid; i < d_head; i += blockDim.x) {
                    s_attn[i] *= inv_d;
                }
            }
        }
    }
    __syncthreads();

    // Step 2: Norm 1
    if (norm_strategy == 0) {
        float val = 0.0f;
        for (int i = tid; i < d_head; i += blockDim.x) {
            float a_val = s_attn[i];
            val += a_val * a_val;
        }

        float sq_sum = block_reduce_sum(val, s_reduce);
        float inv_rms = rsqrtf(sq_sum / (float)d_head + 1e-8f);

        for (int i = tid; i < d_head; i += blockDim.x) {
            s_mid[i] = s_p_in[i] + alpha * (s_attn[i] * inv_rms);
        }
    } else {
        float val = 0.0f;
        for (int i = tid; i < d_head; i += blockDim.x) {
            float m_val = s_p_in[i] + s_attn[i];
            s_mid[i] = m_val;
            val += m_val * m_val;
        }

        float sq_sum = block_reduce_sum(val, s_reduce);
        float norm_val = rsqrtf(sq_sum + 1e-8f);
        float s_scale = sphere_radius * norm_val;

        for (int i = tid; i < d_head; i += blockDim.x) {
            s_mid[i] *= s_scale;
        }
    }
    __syncthreads();

    // Step 3: SwiGLU Activation
    for (int j = tid; j < ffn_dim; j += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            float m_val = s_mid[d];
            gate += m_val * __ldg(w_gate + d * ffn_dim + j);
            up   += m_val * __ldg(w_up + d * ffn_dim + j);
        }
        s_ffn_inter[j] = swiglu(gate, up);
    }
    __syncthreads();

    // Step 4: Down Projection
    for (int d = tid; d < d_head; d += blockDim.x) {
        float ffn_acc = 0.0f;
        for (int j = 0; j < ffn_dim; ++j) {
            ffn_acc += s_ffn_inter[j] * __ldg(w_down + j * d_head + d);
        }
        s_attn[d] = ffn_acc;
    }
    __syncthreads();

    // Step 5: Norm 2 & Write Output
    if (norm_strategy == 0) {
        float val = 0.0f;
        for (int i = tid; i < d_head; i += blockDim.x) {
            float a_v = s_attn[i];
            val += a_v * a_v;
        }

        float ffn_sq = block_reduce_sum(val, s_reduce);
        float inv_ffn_rms = rsqrtf(ffn_sq / (float)d_head + 1e-8f);

        for (int i = tid; i < d_head; i += blockDim.x) {
            float res = s_mid[i] + alpha * (s_attn[i] * inv_ffn_rms);
            curr_out[i] = fminf(fmaxf(res, -100.0f), 100.0f);
        }
    } else {
        float val = 0.0f;
        for (int i = tid; i < d_head; i += blockDim.x) {
            float a_val = s_mid[i] + s_attn[i];
            s_attn[i] = a_val;
            val += a_val * a_val;
        }

        float sq_sum = block_reduce_sum(val, s_reduce);
        float norm_val = rsqrtf(sq_sum + 1e-8f);
        float s_scale = sphere_radius * norm_val;

        for (int i = tid; i < d_head; i += blockDim.x) {
            float res = s_attn[i] * s_scale;
            curr_out[i] = fminf(fmaxf(res, -100.0f), 100.0f);
        }
    }
}

struct MicroBlockDeviceBufferPool {
    float* d_p_in = nullptr;
    float* d_k_cache = nullptr;
    float* d_v_cache = nullptr;
    float* d_w_gate = nullptr;
    float* d_w_up = nullptr;
    float* d_w_down = nullptr;
    float* d_p_out = nullptr;
    size_t pin_cap = 0;
    size_t kv_cap = 0;
    size_t w_cap = 0;
    cudaStream_t last_stream = nullptr;

    ~MicroBlockDeviceBufferPool() {
        if (d_p_in) { cudaFree(d_p_in); d_p_in = nullptr; }
        if (d_k_cache) { cudaFree(d_k_cache); d_k_cache = nullptr; }
        if (d_v_cache) { cudaFree(d_v_cache); d_v_cache = nullptr; }
        if (d_w_gate) { cudaFree(d_w_gate); d_w_gate = nullptr; }
        if (d_w_up) { cudaFree(d_w_up); d_w_up = nullptr; }
        if (d_w_down) { cudaFree(d_w_down); d_w_down = nullptr; }
        if (d_p_out) { cudaFree(d_p_out); d_p_out = nullptr; }
        cudaGetLastError();
    }

    void ensure_capacity(size_t batch_size, size_t d_head, size_t ffn_dim, size_t kv_len, cudaStream_t stream) {
        if (last_stream != stream && last_stream != nullptr) {
            cudaStreamSynchronize(last_stream);
        }
        last_stream = stream;

        size_t pin_elems = batch_size * d_head;
        size_t kv_elems = kv_len * d_head;
        size_t w1_elems = d_head * ffn_dim;
        size_t w2_elems = ffn_dim * d_head;
        size_t max_w_elems = w1_elems > w2_elems ? w1_elems : w2_elems;

        if (pin_elems > pin_cap) {
            if (d_p_in) cudaFreeAsync(d_p_in, stream);
            if (d_p_out) cudaFreeAsync(d_p_out, stream);
            pin_cap = pin_elems * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_p_in, pin_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_p_out, pin_cap * sizeof(float), stream));
        }
        if (kv_elems > kv_cap) {
            if (d_k_cache) cudaFreeAsync(d_k_cache, stream);
            if (d_v_cache) cudaFreeAsync(d_v_cache, stream);
            kv_cap = kv_elems * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_k_cache, kv_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_v_cache, kv_cap * sizeof(float), stream));
        }
        if (max_w_elems > w_cap) {
            if (d_w_gate) cudaFreeAsync(d_w_gate, stream);
            if (d_w_up) cudaFreeAsync(d_w_up, stream);
            if (d_w_down) cudaFreeAsync(d_w_down, stream);
            w_cap = max_w_elems * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_w_gate, w_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_w_up, w_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_w_down, w_cap * sizeof(float), stream));
        }
    }
};

static thread_local MicroBlockDeviceBufferPool g_micro_pool;

extern "C" void launch_fused_micro_block(
    const float* p_in,
    const float* k_cache,
    const float* v_cache,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* p_out,
    int batch_size,
    int d_head,
    int ffn_dim,
    int kv_len,
    int norm_strategy,
    float alpha,
    float sphere_radius,
    cudaStream_t stream
) {
    if (batch_size <= 0 || d_head <= 0 || ffn_dim <= 0) return;

    g_micro_pool.ensure_capacity(batch_size, d_head, ffn_dim, kv_len, stream);

    bool was_host_copied = false;
    const float* dev_pin = get_device_ptr(p_in, g_micro_pool.d_p_in, batch_size * d_head, stream, &was_host_copied);
    const float* dev_k = get_device_ptr(k_cache, g_micro_pool.d_k_cache, kv_len * d_head, stream, &was_host_copied);
    const float* dev_v = get_device_ptr(v_cache, g_micro_pool.d_v_cache, kv_len * d_head, stream, &was_host_copied);
    const float* dev_wgate = get_device_ptr(w_gate, g_micro_pool.d_w_gate, d_head * ffn_dim, stream, &was_host_copied);
    const float* dev_wup = get_device_ptr(w_up, g_micro_pool.d_w_up, d_head * ffn_dim, stream, &was_host_copied);
    const float* dev_wdown = get_device_ptr(w_down, g_micro_pool.d_w_down, ffn_dim * d_head, stream, &was_host_copied);

    float* dev_pout = get_device_ptr_mut(p_out, g_micro_pool.d_p_out, batch_size * d_head, stream, &was_host_copied);

    int threads_per_block = 128;
    int blocks = batch_size;
    int dh_pad = (d_head + 3) & ~3;
    int ffn_pad = (ffn_dim + 3) & ~3;
    int max_warps = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    int max_warps_pad = (max_warps + 3) & ~3;

    size_t shared_mem_bytes = (3 * dh_pad + ffn_pad + max_warps_pad) * sizeof(float);

    fused_micro_block_kernel<<<blocks, threads_per_block, shared_mem_bytes, stream>>>(
        dev_pin, dev_k, dev_v, dev_wgate, dev_wup, dev_wdown, dev_pout,
        batch_size, d_head, ffn_dim, kv_len, norm_strategy, alpha, sphere_radius
    );

    bool sync_needed = copy_back_if_host(p_out, dev_pout, batch_size * d_head, stream);
    if (sync_needed || was_host_copied) {
        cudaStreamSynchronize(stream);
    }
}
