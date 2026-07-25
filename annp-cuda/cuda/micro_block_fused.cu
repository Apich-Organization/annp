#include "common.cuh"

/**
 * Fused CUDA Kernel for Micro-Block Computing Unit.
 * Performs:
 * 1. Local Attention against FIFO KV Cache: p_attn = Softmax(p_in * K^T) * V
 * 2. MicroNorm 1 (Micro-RMSNorm with alpha scaling OR Sphere Normalization)
 * 3. Local SwiGLU FFN (8x d_head expansion): p_ffn = (p_mid * W_gate * Swish) * W_up * W_down
 * 4. MicroNorm 2
 */
extern "C" __global__ void fused_micro_block_kernel(
    const float* __restrict__ p_in,           // [batch_size, d_head]
    const float* __restrict__ k_cache,        // [kv_len, d_head]
    const float* __restrict__ v_cache,        // [kv_len, d_head]
    const float* __restrict__ w_gate,         // [d_head, ffn_dim]
    const float* __restrict__ w_up,           // [d_head, ffn_dim]
    const float* __restrict__ w_down,         // [ffn_dim, d_head]
    float* __restrict__ p_out,                // [batch_size, d_head]
    int batch_size,
    int d_head,
    int ffn_dim,                              // 8 * d_head
    int kv_len,
    int norm_strategy,                        // 0: MicroRMSNorm, 1: SphereNorm
    float alpha,
    float sphere_radius
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= batch_size) return;

    const float* curr_p = p_in + pid * d_head;
    float* curr_out = p_out + pid * d_head;

    extern __shared__ float s_mem[];
    // Memory offsets within Shared SRAM
    float* s_attn = s_mem;                                 // [d_head]
    float* s_mid = s_mem + d_head;                          // [d_head]
    float* s_ffn_intermediate = s_mem + 2 * d_head;         // [ffn_dim]

    // Step 1: Local Attention dot products
    float scale = 1.0f / sqrtf((float)d_head);
    
    // Clear s_attn accumulator
    for (int d = 0; d < d_head; ++d) {
        s_attn[d] = 0.0f;
    }

    if (kv_len > 0) {
        float max_score = -1e9f;
        
        // Compute dot products and max score for Softmax stability
        for (int k = 0; k < kv_len; ++k) {
            float score = 0.0f;
            for (int d = 0; d < d_head; ++d) {
                score += curr_p[d] * k_cache[k * d_head + d];
            }
            score *= scale;
            if (score > max_score) max_score = score;
        }

        // Softmax denominator and weighted V accumulation
        float sum_exp = 0.0f;
        for (int k = 0; k < kv_len; ++k) {
            float score = 0.0f;
            for (int d = 0; d < d_head; ++d) {
                score += curr_p[d] * k_cache[k * d_head + d];
            }
            score = expf(score * scale - max_score);
            sum_exp += score;

            for (int d = 0; d < d_head; ++d) {
                s_attn[d] += score * v_cache[k * d_head + d];
            }
        }

        if (sum_exp > 1e-8f) {
            for (int d = 0; d < d_head; ++d) {
                s_attn[d] /= sum_exp;
            }
        }
    }

    // Step 2: MicroNorm 1 + Residual
    if (norm_strategy == 0) {
        // Micro-RMSNorm with alpha scaling
        float sq_sum = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            sq_sum += s_attn[d] * s_attn[d];
        }
        float rms = sqrtf(sq_sum / (float)d_head + 1e-8f);
        for (int d = 0; d < d_head; ++d) {
            s_mid[d] = curr_p[d] + alpha * (s_attn[d] / rms);
        }
    } else {
        // Sphere Normalization
        float sq_sum = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            float val = curr_p[d] + s_attn[d];
            s_mid[d] = val;
            sq_sum += val * val;
        }
        float norm = sqrtf(sq_sum + 1e-8f);
        float s_scale = sphere_radius / norm;
        for (int d = 0; d < d_head; ++d) {
            s_mid[d] *= s_scale;
        }
    }

    // Step 3: Local SwiGLU FFN (8x d_head expansion)
    for (int j = 0; j < ffn_dim; ++j) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            gate += s_mid[d] * w_gate[d * ffn_dim + j];
            up += s_mid[d] * w_up[d * ffn_dim + j];
        }
        s_ffn_intermediate[j] = swiglu(gate, up);
    }

    // Down projection to d_head
    for (int d = 0; d < d_head; ++d) {
        float ffn_out_d = 0.0f;
        for (int j = 0; j < ffn_dim; ++j) {
            ffn_out_d += s_ffn_intermediate[j] * w_down[j * d_head + d];
        }

        // Step 4: MicroNorm 2 + Residual Output
        if (norm_strategy == 0) {
            curr_out[d] = s_mid[d] + alpha * ffn_out_d;
        } else {
            float sum_val = s_mid[d] + ffn_out_d;
            curr_out[d] = sum_val;
        }
    }

    if (norm_strategy == 1) {
        float sq_sum = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            sq_sum += curr_out[d] * curr_out[d];
        }
        float norm = sqrtf(sq_sum + 1e-8f);
        float s_scale = sphere_radius / norm;
        for (int d = 0; d < d_head; ++d) {
            curr_out[d] *= s_scale;
        }
    }
}

// C-FFI Wrapper Function
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
    int threads_per_block = 64;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_bytes = (2 * d_head + ffn_dim) * sizeof(float);

    fused_micro_block_kernel<<<blocks, threads_per_block, shared_mem_bytes, stream>>>(
        p_in, k_cache, v_cache, w_gate, w_up, w_down, p_out,
        batch_size, d_head, ffn_dim, kv_len, norm_strategy, alpha, sphere_radius
    );
}
