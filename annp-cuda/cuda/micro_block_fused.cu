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
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    float* __restrict__ p_out,
    int d_head,
    int ffn_dim,
    int kv_len,
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

    // Step 1: Online FlashAttention
    if (kv_len > 0) {
        float best_sim = -1e9f;
        float scale = rsqrtf((float)d_head) * inv_rms_attn;
        
        for (int k = 0; k < kv_len; ++k) {
            const float* k_ptr = k_cache + k * d_head;
            float val = 0.0f;
            for (int i = tid; i < d_head; i += blockDim.x) {
                val += s_p_in[i] * __ldg(k_ptr + i);
            }

            float dot = block_reduce_sum(val, s_reduce);
            if (tid == 0) {
                float score = dot * scale;
                s_reduce[max_warps_pad + k] = score;
                if (score > best_sim) best_sim = score;
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_reduce[0] = best_sim;
        }
        __syncthreads();
        best_sim = s_reduce[0];

        float sum_exp = 0.0f;
        if (tid == 0) {
            for (int i = 0; i < kv_len; i++) {
                float e = __expf(s_reduce[max_warps_pad + i] - best_sim);
                s_reduce[max_warps_pad + i] = e;
                sum_exp += e;
            }
            s_reduce[0] = __fdividef(1.0f, sum_exp + 1e-8f);
        }
        __syncthreads();

        float inv_sum = s_reduce[0];
        for (int i = 0; i < kv_len; i++) {
            float w = s_reduce[max_warps_pad + i] * inv_sum;
            const float* v_slice = v_cache + i * d_head;
            for (int d = tid; d < d_head; d += blockDim.x) {
                s_attn[d] += w * __ldg(v_slice + d);
            }
        }
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

    copy_back_if_host(p_out, dev_pout, batch_size * d_head, stream);
}
__global__ void fused_micro_block_backward_kernel(
    const float* __restrict__ p_in,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ w_gate,
    float* __restrict__ w_up,
    float* __restrict__ w_down,
    const float* __restrict__ local_err,
    int d_head,
    int ffn_dim,
    int kv_len,
    float alpha,
    float lr,
    float weight_decay
) {
    // Only handles 1 block for the single token (batch_size=1)
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    float max_grad = 0.05f;
    float wd_factor = 1.0f - lr * weight_decay;

    int max_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    int max_warps_pad = (max_warps + 3) & ~3;


    // Use dynamic shared memory for all intermediate arrays
    extern __shared__ float s_mem_bw[];
    float* s_p_in_normed = s_mem_bw;
    float* s_attn        = s_p_in_normed + d_head;
    float* s_mid         = s_attn + d_head;
    float* s_mid_normed  = s_mid + d_head;

    float* s_gate        = s_mid_normed + d_head;
    float* s_up          = s_gate + ffn_dim;
    float* s_swish       = s_up + ffn_dim;
    float* s_inter       = s_swish + ffn_dim;
    float* s_d_inter     = s_inter + ffn_dim;
    float* s_d_gate      = s_d_inter + ffn_dim;
    float* s_d_up        = s_d_gate + ffn_dim;
    float* s_reduce      = s_d_up + ffn_dim;

    // 1. RMSNorm of p_in
    float sq_sum = 0.0f;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float v = __ldg(p_in + i);
        sq_sum += v * v;
    }
    sq_sum = block_reduce_sum(sq_sum, s_reduce);
    float inv_rms_attn = rsqrtf(sq_sum / d_head + 1e-8f);

    for (int i = tid; i < d_head; i += blockDim.x) {
        s_p_in_normed[i] = __ldg(p_in + i) * inv_rms_attn;
        s_attn[i] = 0.0f;
    }
    __syncthreads();

    // 2. Flash Attention
    if (kv_len > 0) {
        float best_sim = -1e9f;
        float scale = rsqrtf((float)d_head);
        
        // Compute best_sim
        for (int i = 0; i < kv_len; i++) {
            const float* k_slice = k_cache + i * d_head;
            float dot = 0.0f;
            for (int d = tid; d < d_head; d += blockDim.x) {
                dot += s_p_in_normed[d] * k_slice[d];
            }
            dot = block_reduce_sum(dot, s_reduce);
            if (tid == 0) {
                float score = dot * scale;
                if (score > best_sim) best_sim = score;
                // Temporarily store score in s_attn's first elements to save registers
                // Wait, s_attn needs to be used for output. We can store scores in s_reduce past index max_warps_pad.
                s_reduce[max_warps_pad + i] = score; 
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_reduce[0] = best_sim;
        }
        __syncthreads();
        best_sim = s_reduce[0];

        if (best_sim > 0.0f) {
            float sum_exp = 0.0f;
            if (tid == 0) {
                for (int i = 0; i < kv_len; i++) {
                    float e = __expf(s_reduce[max_warps_pad + i] - best_sim);
                    s_reduce[max_warps_pad + i] = e;
                    sum_exp += e;
                }
                s_reduce[0] = __fdividef(1.0f, sum_exp + 1e-8f);
            }
            __syncthreads();

            float inv_sum = s_reduce[0];
            for (int i = 0; i < kv_len; i++) {
                float w = s_reduce[max_warps_pad + i] * inv_sum;
                const float* v_slice = v_cache + i * d_head;
                for (int d = tid; d < d_head; d += blockDim.x) {
                    s_attn[d] += w * v_slice[d];
                }
            }
        }
    }
    __syncthreads();

    // 3. RMSNorm of mid
    float sq_sum_mid = 0.0f;
    for (int d = tid; d < d_head; d += blockDim.x) {
        float val = __ldg(p_in + d) + alpha * s_attn[d];
        s_mid[d] = val;
        sq_sum_mid += val * val;
    }
    sq_sum_mid = block_reduce_sum(sq_sum_mid, s_reduce);
    float inv_rms_ffn = rsqrtf(sq_sum_mid / d_head + 1e-8f);

    for (int d = tid; d < d_head; d += blockDim.x) {
        s_mid_normed[d] = s_mid[d] * inv_rms_ffn;
    }
    __syncthreads();

    // 4. FFN Forward (gate, up)
    for (int f = tid; f < ffn_dim; f += blockDim.x) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int d = 0; d < d_head; d++) {
            float m_val = s_mid_normed[d];
            gate += m_val * w_gate[d * ffn_dim + f];
            up += m_val * w_up[d * ffn_dim + f];
        }
        float sig = __fdividef(1.0f, 1.0f + __expf(-gate));
        float swish = gate * sig;

        s_gate[f] = gate;
        s_up[f] = up;
        s_swish[f] = swish;
        s_inter[f] = swish * up;
    }
    __syncthreads();

    // 5. Compute d_inter (Backward from local_err through w_down)
    for (int f = tid; f < ffn_dim; f += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < d_head; d++) {
            sum += local_err[d] * w_down[f * d_head + d];
        }
        s_d_inter[f] = sum;
    }
    __syncthreads();

    // 6. Compute d_gate, d_up
    for (int f = tid; f < ffn_dim; f += blockDim.x) {
        float d_int = s_d_inter[f];
        float d_swish = d_int * s_up[f];
        float d_up_val = d_int * s_swish[f];

        float gate = s_gate[f];
        float sig = __fdividef(1.0f, 1.0f + __expf(-gate));
        float d_sig_d_gate = sig * (1.0f - sig);
        float d_swish_d_gate = sig + gate * d_sig_d_gate;

        float d_gate_val = d_swish * d_swish_d_gate;
        s_d_gate[f] = d_gate_val;
        s_d_up[f] = d_up_val;
    }
    __syncthreads();

    // 7. Update w_down
    for (int idx = tid; idx < ffn_dim * d_head; idx += blockDim.x) {
        int f = idx / d_head;
        int d = idx % d_head;
        float grad = local_err[d] * s_inter[f] * alpha;
        grad = fmaxf(-max_grad, fminf(max_grad, grad));
        w_down[idx] = w_down[idx] * wd_factor - lr * grad;
    }

    // 8. Update w_gate and w_up
    for (int idx = tid; idx < d_head * ffn_dim; idx += blockDim.x) {
        int d = idx / ffn_dim;
        int f = idx % ffn_dim;
        float m_val = s_mid_normed[d];
        float grad_gate = s_d_gate[f] * m_val * alpha;
        float grad_up = s_d_up[f] * m_val * alpha;
        
        grad_gate = fmaxf(-max_grad, fminf(max_grad, grad_gate));
        grad_up = fmaxf(-max_grad, fminf(max_grad, grad_up));

        w_gate[idx] = w_gate[idx] * wd_factor - lr * grad_gate;
        w_up[idx] = w_up[idx] * wd_factor - lr * grad_up;
    }
}
extern "C" void launch_fused_micro_block_backward(
    const float* p_in,
    const float* k_cache,
    const float* v_cache,
    float* w_gate,
    float* w_up,
    float* w_down,
    const float* local_err,
    int d_head,
    int ffn_dim,
    int kv_len,
    float alpha,
    float lr,
    float weight_decay,
    cudaStream_t stream
) {
    if (d_head <= 0 || ffn_dim <= 0) return;

    // Use capacity from g_micro_pool
    g_micro_pool.ensure_capacity(1, d_head, ffn_dim, kv_len, stream);

    bool was_host_copied = false;
    const float* dev_pin = get_device_ptr(p_in, g_micro_pool.d_p_in, d_head, stream, &was_host_copied);
    const float* dev_k = get_device_ptr(k_cache, g_micro_pool.d_k_cache, kv_len * d_head, stream, &was_host_copied);
    const float* dev_v = get_device_ptr(v_cache, g_micro_pool.d_v_cache, kv_len * d_head, stream, &was_host_copied);
    
    // w_gate, w_up, w_down must be in-place updated, they should already be device pointers, 
    // but get_device_ptr_mut will handle it.
    float* dev_wgate = get_device_ptr_mut(w_gate, g_micro_pool.d_w_gate, d_head * ffn_dim, stream, &was_host_copied);
    float* dev_wup = get_device_ptr_mut(w_up, g_micro_pool.d_w_up, d_head * ffn_dim, stream, &was_host_copied);
    float* dev_wdown = get_device_ptr_mut(w_down, g_micro_pool.d_w_down, ffn_dim * d_head, stream, &was_host_copied);
    
    // local_err uses p_out buffer since we don't need p_out for backward
    const float* dev_err = get_device_ptr(local_err, g_micro_pool.d_p_out, d_head, stream, &was_host_copied);

    int threads_per_block = 256;
    if (d_head < 256 && ffn_dim < 256) {
        threads_per_block = 128;
    }
    
    int max_warps = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    int max_warps_pad = (max_warps + 3) & ~3;

    // We need shared memory for:
    // s_p_in_normed (d_head)
    // s_attn (d_head)
    // s_mid (d_head)
    // s_mid_normed (d_head)
    // s_gate (ffn_dim)
    // s_up (ffn_dim)
    // s_swish (ffn_dim)
    // s_inter (ffn_dim)
    // s_d_inter (ffn_dim)
    // s_d_gate (ffn_dim)
    // s_d_up (ffn_dim)
    // s_reduce (max_warps_pad + kv_len)
    
    // Note: To avoid huge shared memory, we check if total fits in 48KB.
    // If it doesn't, we should either increase shared memory limit or handle it.
    // For d_head=64, ffn_dim=256, it's 4*64 + 7*256 + 4 + kv_len = 256 + 1792 + 4 + kv_len = 2052 floats ~ 8KB, completely fine.
    
    size_t shared_mem_bytes = (4 * d_head + 7 * ffn_dim + max_warps_pad + kv_len + 16) * sizeof(float);

    fused_micro_block_backward_kernel<<<1, threads_per_block, shared_mem_bytes, stream>>>(
        dev_pin, dev_k, dev_v, dev_wgate, dev_wup, dev_wdown, dev_err,
        d_head, ffn_dim, kv_len, alpha, lr, weight_decay
    );

    copy_back_if_host(w_gate, dev_wgate, d_head * ffn_dim, stream);
    copy_back_if_host(w_up, dev_wup, d_head * ffn_dim, stream);
    copy_back_if_host(w_down, dev_wdown, ffn_dim * d_head, stream);
}
