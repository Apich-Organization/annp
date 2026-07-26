#include "common.cuh"
#include <cuda_runtime.h>

// Block-wide reduction using warp shuffle + shared memory across warps
__device__ inline float block_reduce_sum(float val, float* s_reduce) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    // 1. Intra-warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Write warp sums to shared memory
    if (lane == 0) {
        s_reduce[wid] = val;
    }
    __syncthreads();

    // 2. Reduce warp sums using the first warp
    int num_warps = (blockDim.x + 31) / 32;
    val = (tid < num_warps) ? s_reduce[lane] : 0.0f;

    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    // Broadcast final result from thread 0 to all threads in block
    if (tid == 0) {
        s_reduce[0] = val;
    }
    __syncthreads();

    return s_reduce[0];
}

__device__ __forceinline__ float swiglu(float gate, float up) {
    float sig = 1.0f / (1.0f + __expf(-gate));
    return (gate * sig) * up;
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

    // Dynamic Shared Memory Allocation
    // Total required size on host launch: (3 * dh_pad + ffn_pad + 32) * sizeof(float)
    extern __shared__ __align__(16) float s_mem[];
    float* s_p_in       = s_mem;
    float* s_attn       = s_mem + dh_pad;
    float* s_mid        = s_mem + 2 * dh_pad;
    float* s_ffn_inter  = s_mem + 3 * dh_pad;
    float* s_reduce     = s_mem + 3 * dh_pad + ffn_pad; // Buffer for block reductions (32 floats)

    // Step 0: Vectorized Load p_in into Shared Memory
    int vec_d_head = d_head / 4;
    int rem_d_head = d_head % 4;

    const float4* curr_p_vec = reinterpret_cast<const float4*>(curr_p);
    float4* s_p_in_vec = reinterpret_cast<float4*>(s_p_in);
    float4* s_attn_vec = reinterpret_cast<float4*>(s_attn);

    for (int i = tid; i < vec_d_head; i += blockDim.x) {
        s_p_in_vec[i] = __ldg(curr_p_vec + i);
        s_attn_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
        s_p_in[i] = __ldg(curr_p + i);
        s_attn[i] = 0.0f;
    }
    __syncthreads();

    // Step 1: Online FlashAttention
    if (kv_len > 0) {
        float m_prev = -1e9f;
        float d_prev = 0.0f;

        for (int k = 0; k < kv_len; ++k) {
            const float* k_ptr = k_cache + k * d_head;
            const float4* k_ptr_vec = reinterpret_cast<const float4*>(k_ptr);

            float val = 0.0f;
            for (int i = tid; i < vec_d_head; i += blockDim.x) {
                float4 p_v = s_p_in_vec[i];
                float4 k_v = __ldg(k_ptr_vec + i);
                val += p_v.x * k_v.x + p_v.y * k_v.y + p_v.z * k_v.z + p_v.w * k_v.w;
            }
            for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
                val += s_p_in[i] * __ldg(k_ptr + i);
            }

            float dot = block_reduce_sum(val, s_reduce);
            float score = dot * rsqrtf((float)d_head);

            float m_curr = fmaxf(m_prev, score);
            float alpha_scale = __expf(m_prev - m_curr);
            float p_val = __expf(score - m_curr);

            d_prev = d_prev * alpha_scale + p_val;
            m_prev = m_curr;

            const float* v_ptr = v_cache + k * d_head;
            const float4* v_ptr_vec = reinterpret_cast<const float4*>(v_ptr);

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
            for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
                s_attn[i] = s_attn[i] * alpha_scale + p_val * __ldg(v_ptr + i);
            }
        }

        if (d_prev > 1e-8f) {
            float inv_d = 1.0f / d_prev;
            for (int i = tid; i < vec_d_head; i += blockDim.x) {
                float4 a_v = s_attn_vec[i];
                s_attn_vec[i] = make_float4(a_v.x * inv_d, a_v.y * inv_d, a_v.z * inv_d, a_v.w * inv_d);
            }
            for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
                s_attn[i] *= inv_d;
            }
        }
    }
    __syncthreads();

    // Step 2: Norm 1
    float4* s_mid_vec = reinterpret_cast<float4*>(s_mid);
    if (norm_strategy == 0) {
        float val = 0.0f;
        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 a_v = s_attn_vec[i];
            val += a_v.x * a_v.x + a_v.y * a_v.y + a_v.z * a_v.z + a_v.w * a_v.w;
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
            val += s_attn[i] * s_attn[i];
        }

        float sq_sum = block_reduce_sum(val, s_reduce);
        float inv_rms = rsqrtf(sq_sum / (float)d_head + 1e-8f);

        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 p_v = s_p_in_vec[i];
            float4 a_v = s_attn_vec[i];
            s_mid_vec[i] = make_float4(
                p_v.x + alpha * (a_v.x * inv_rms),
                p_v.y + alpha * (a_v.y * inv_rms),
                p_v.z + alpha * (a_v.z * inv_rms),
                p_v.w + alpha * (a_v.w * inv_rms)
            );
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
            s_mid[i] = s_p_in[i] + alpha * (s_attn[i] * inv_rms);
        }
    } else {
        float val = 0.0f;
        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 p_v = s_p_in_vec[i];
            float4 a_v = s_attn_vec[i];
            float4 m_v = make_float4(p_v.x + a_v.x, p_v.y + a_v.y, p_v.z + a_v.z, p_v.w + a_v.w);
            s_mid_vec[i] = m_v;
            val += m_v.x * m_v.x + m_v.y * m_v.y + m_v.z * m_v.z + m_v.w * m_v.w;
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
            float m_val = s_p_in[i] + s_attn[i];
            s_mid[i] = m_val;
            val += m_val * m_val;
        }

        float sq_sum = block_reduce_sum(val, s_reduce);
        float norm_val = rsqrtf(sq_sum + 1e-8f);
        float s_scale = sphere_radius * norm_val;

        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 m_v = s_mid_vec[i];
            s_mid_vec[i] = make_float4(m_v.x * s_scale, m_v.y * s_scale, m_v.z * s_scale, m_v.w * s_scale);
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
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
    float4* curr_out_vec = reinterpret_cast<float4*>(curr_out);
    if (norm_strategy == 0) {
        float val = 0.0f;
        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 a_v = s_attn_vec[i];
            val += a_v.x * a_v.x + a_v.y * a_v.y + a_v.z * a_v.z + a_v.w * a_v.w;
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
            val += s_attn[i] * s_attn[i];
        }

        float ffn_sq = block_reduce_sum(val, s_reduce);
        float inv_ffn_rms = rsqrtf(ffn_sq / (float)d_head + 1e-8f);

        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 m_v = s_mid_vec[i];
            float4 a_v = s_attn_vec[i];
            float4 res = make_float4(
                fminf(fmaxf(m_v.x + alpha * (a_v.x * inv_ffn_rms), -100.0f), 100.0f),
                fminf(fmaxf(m_v.y + alpha * (a_v.y * inv_ffn_rms), -100.0f), 100.0f),
                fminf(fmaxf(m_v.z + alpha * (a_v.z * inv_ffn_rms), -100.0f), 100.0f),
                fminf(fmaxf(m_v.w + alpha * (a_v.w * inv_ffn_rms), -100.0f), 100.0f)
            );
            curr_out_vec[i] = res;
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
            float res = s_mid[i] + alpha * (s_attn[i] * inv_ffn_rms);
            curr_out[i] = fminf(fmaxf(res, -100.0f), 100.0f);
        }
    } else {
        float val = 0.0f;
        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 m_v = s_mid_vec[i];
            float4 a_v = s_attn_vec[i];
            float4 sum_v = make_float4(m_v.x + a_v.x, m_v.y + a_v.y, m_v.z + a_v.z, m_v.w + a_v.w);
            s_attn_vec[i] = sum_v;
            val += sum_v.x * sum_v.x + sum_v.y * sum_v.y + sum_v.z * sum_v.z + sum_v.w * sum_v.w;
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
            float a_val = s_mid[i] + s_attn[i];
            s_attn[i] = a_val;
            val += a_val * a_val;
        }

        float sq_sum = block_reduce_sum(val, s_reduce);
        float norm_val = rsqrtf(sq_sum + 1e-8f);
        float s_scale = sphere_radius * norm_val;

        for (int i = tid; i < vec_d_head; i += blockDim.x) {
            float4 a_v = s_attn_vec[i];
            curr_out_vec[i] = make_float4(a_v.x * s_scale, a_v.y * s_scale, a_v.z * s_scale, a_v.w * s_scale);
        }
        for (int i = vec_d_head * 4 + tid; i < d_head; i += blockDim.x) {
            curr_out[i] = s_attn[i] * s_scale;
        }
    }
}
