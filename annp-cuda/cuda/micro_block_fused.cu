#include "common.cuh"
#include <cuda_runtime.h>

// Block-wide reduction using warp shuffle + shared memory across warps
__device__ inline float block_reduce_sum(float val, float* s_reduce) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    // 1. Intra-warp reduction
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
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
        unsigned int mask = __activemask();
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
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
    const float* __restrict__ fast_weight,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    float* __restrict__ p_out,
    int d_head,
    int ffn_dim,
    int norm_strategy,
    float alpha,
    float sphere_radius
) {
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    
    extern __shared__ float s_mem[];
    float* s_p_in  = s_mem;
    float* s_attn  = s_p_in + d_head;
    float* s_mid   = s_attn + d_head;
    float* s_ffn_inter = s_mid + d_head;
    int max_warps_pad = ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE + 3) & ~3;
    float* s_reduce = s_ffn_inter + ffn_dim;
    
    const float* curr_p = p_in;
    float* curr_out = p_out;

    // Step 0: Load p_in into Shared Memory & initialize s_attn
    float sq_sum_attn = 0.0f;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float p_v = __ldg(curr_p + i);
        s_p_in[i] = p_v;
        sq_sum_attn += p_v * p_v;
        s_attn[i] = 0.0f;
    }
    
    sq_sum_attn = block_reduce_sum(sq_sum_attn, s_reduce);
    float inv_rms_attn = rsqrtf(sq_sum_attn / (float)d_head + 1e-8f);
    __syncthreads();

    // Step 1: Fast Weight Implicit Memory
    // attn_out[r] = sum_c(fast_weight[r * d_head + c] * p_in_normed[c])
    // Wait, s_p_in is NOT normalized yet. But p_in (which is last_p_in) is normalized here.
    // Let's normalize it first.
    // inv_rms_attn is already computed above.
    for (int i = tid; i < d_head; i += blockDim.x) {
        s_attn[i] = 0.0f;
    }
    __syncthreads();

    // Matrix-vector multiplication: fast_weight (d_head x d_head) * (s_p_in * inv_rms_attn)
    for (int r = tid; r < d_head; r += blockDim.x) {
        float sum = 0.0f;
        for (int c = 0; c < d_head; ++c) {
            float normed_in = s_p_in[c] * inv_rms_attn;
            sum += __ldg(fast_weight + r * d_head + c) * normed_in;
        }
        s_attn[r] = sum;
    }
    __syncthreads();

    // Step 2: Additive Residual 1 and Pre-Norm for FFN
    float sq_sum_mid = 0.0f;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float val = s_p_in[i] + alpha * s_attn[i];
        s_mid[i] = val;
        sq_sum_mid += val * val;
    }

    sq_sum_mid = block_reduce_sum(sq_sum_mid, s_reduce);
    float inv_rms_ffn = rsqrtf(sq_sum_mid / (float)d_head + 1e-8f);
    __syncthreads();

    // Step 3: SwiGLU Activation
    for (int j = tid; j < ffn_dim; j += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            float m_val = s_mid[d] * inv_rms_ffn; // Pre-norm
            gate += m_val * __ldg(w_gate + d * ffn_dim + j);
            up   += m_val * __ldg(w_up + d * ffn_dim + j);
        }
        float sig = __fdividef(1.0f, 1.0f + __expf(-gate));
        float swish = gate * sig;
        s_ffn_inter[j] = swish * up;
    }
    __syncthreads();

    // Step 4: Down Projection
    for (int d = tid; d < d_head; d += blockDim.x) {
        float ffn_acc = 0.0f;
        for (int j = 0; j < ffn_dim; ++j) {
            ffn_acc += s_ffn_inter[j] * __ldg(w_down + j * d_head + d);
        }
        s_attn[d] = ffn_acc; // Reuse s_attn for down projection output
    }
    __syncthreads();

    // Step 5: Additive Residual 2 & Output Clipping
    for (int i = tid; i < d_head; i += blockDim.x) {
        float res = s_mid[i] + alpha * s_attn[i];
        curr_out[i] = fminf(fmaxf(res, -100.0f), 100.0f);
    }
}

struct MicroBlockDeviceBufferPool {
    float* d_p_in = nullptr;
    float* d_fast_weight = nullptr;
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
        if (d_fast_weight) { cudaFree(d_fast_weight); d_fast_weight = nullptr; }
                if (d_w_gate) { cudaFree(d_w_gate); d_w_gate = nullptr; }
        if (d_w_up) { cudaFree(d_w_up); d_w_up = nullptr; }
        if (d_w_down) { cudaFree(d_w_down); d_w_down = nullptr; }
        if (d_p_out) { cudaFree(d_p_out); d_p_out = nullptr; }
        cudaGetLastError();
    }

    void ensure_capacity(size_t batch_size, size_t d_head, size_t ffn_dim, cudaStream_t stream) {
        last_stream = stream;

        size_t pin_elems = batch_size * d_head;
        size_t fw_elems = d_head * d_head;
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
        if (fw_elems > kv_cap) {
            if (d_fast_weight) cudaFreeAsync(d_fast_weight, stream);
            kv_cap = fw_elems * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_fast_weight, kv_cap * sizeof(float), stream));
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
    const float* fast_weight,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* p_out,
    int batch_size,
    int d_head,
    int ffn_dim,
    int norm_strategy,
    float alpha,
    float sphere_radius,
    cudaStream_t stream
) {
    if (batch_size <= 0 || d_head <= 0 || ffn_dim <= 0) return;

    g_micro_pool.ensure_capacity(batch_size, d_head, ffn_dim, stream);

    bool was_host_copied = false;
    const float* dev_pin = get_device_ptr(p_in, g_micro_pool.d_p_in, batch_size * d_head, stream, &was_host_copied);
    const float* dev_fast_weight = get_device_ptr(fast_weight, g_micro_pool.d_fast_weight, d_head * d_head, stream, &was_host_copied);
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
        dev_pin, dev_fast_weight, dev_wgate, dev_wup, dev_wdown, dev_pout,
        d_head, ffn_dim, norm_strategy, alpha, sphere_radius
    );

    copy_back_if_host(p_out, dev_pout, batch_size * d_head, stream);
}
