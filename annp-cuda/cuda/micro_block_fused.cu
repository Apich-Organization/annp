// annp-cuda/cuda/micro_block_fused.cu
//
// GPU inference kernel for a single MicroBlockNode forward pass.
// This is the INFERENCE-ONLY accelerated path. Training (backprop) is CPU-only.
//
// ═══════════════════════════════════════════════════════════════════════════════
// CPU / GPU ALIGNMENT — verified against micro_block.rs process_sub_batch()
// ═══════════════════════════════════════════════════════════════════════════════
//
// The kernel implements exactly the inference portion of MicroBlockNode::process_sub_batch:
//
//   Step 0: Aggregate batch → p_in (CPU averages particles; GPU gets pre-averaged p_in)
//   Step 1: attn_out = fast_weight · RMSNorm(p_in)          [CPU lines 664-671]
//   Step 2: s_mid = p_in + alpha * attn_out                 [CPU lines 678-684]
//   Step 3: SwiGLU FFN: h = swish(W_gate · RMSNorm(s_mid)) [CPU lines 693-708]
//             ⊙ W_up · RMSNorm(s_mid)
//   Step 4: down_proj = W_down · h                          [CPU lines 705-707]
//   Step 5: p_out = s_mid + alpha * down_proj               [CPU lines 710-716]
//
// RMSNorm formula (both CPU and GPU):  x_norm = x / sqrt(mean(x²) + ε),  ε = 1e-8
// alpha: per-subnode scale factor (Thompson-sampled winning subnode's alpha)
//
// IMPORTANT: The GPU kernel does NOT implement:
//   - Particle credit computation (ΔR = R_after - R_before): done on CPU after kernel
//   - Hebbian fast_weight update: CPU-only (requires full backprop through RMSNorm)
//   - FFN weight update (w_gate, w_up, w_down gradient steps): CPU-only
//   - TD learning (last_prediction update, last_token_id tracking): CPU-only
//
// WHY GPU ONLY FOR INFERENCE?
//   Backprop requires storing intermediate activations (gate_arr, up_arr, swish_arr,
//   s_mid_normed, d_gate_arr, d_up_arr, d_s_mid_normed, dot_product) that would
//   require significant shared memory or global memory bandwidth. The CPU already
//   handles this efficiently for the current batch sizes. GPU is worth it only for
//   the matvec and activation compute in the forward path.
//
// OUTPUT CLIPPING [-100, 100]:
//   Applied in GPU (Step 5) but absent in CPU path. This is a GPU-specific numerical
//   guard that prevents NaN propagation in very early training when weights are random.
//   The clip is wide enough (±100) not to affect normal operation.
//
// ═══════════════════════════════════════════════════════════════════════════════
// SHARED MEMORY LAYOUT (per block, all f32):
//   s_p_in      [d_head]       — input particle embedding
//   s_attn      [d_head]       — fast_weight output (reused for down-proj output in Step 4)
//   s_mid       [d_head]       — intermediate: p_in + alpha * attn_out
//   s_ffn_inter [ffn_dim]      — SwiGLU intermediate activations
//   s_reduce    [max_warps_pad] — warp-level reduction scratch
// Total: (3*d_head_pad + ffn_dim_pad + max_warps_pad) * 4 bytes
// ═══════════════════════════════════════════════════════════════════════════════

#include "common.cuh"
#include <cuda_runtime.h>

// Block-wide sum reduction using warp shuffle + one round of inter-warp reduction.
//
// Result is broadcast to all threads via s_reduce[0] after __syncthreads().
// WHY NOT just use warp_reduce_sum from common.cuh?
//   warp_reduce_sum only reduces within one warp. For block-wide reduction we need
//   to combine across all warps. This function handles the full two-level reduction.
__device__ inline float block_reduce_sum(float val, float* s_reduce) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // Level 1: reduce within each warp
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }

    // Lane 0 of each warp writes its partial sum to shared memory
    if (lane == 0) {
        s_reduce[wid] = val;
    }
    __syncthreads();

    // Level 2: first warp reduces across all warp sums
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (tid < num_warps) ? s_reduce[tid] : 0.0f;

    if (wid == 0) {
        unsigned int mask = __activemask();
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
    }

    // Thread 0 writes result; __syncthreads lets all threads read it
    if (tid == 0) {
        s_reduce[0] = val;
    }
    __syncthreads();

    return s_reduce[0];
}

// ─── Main Kernel: 1 block per particle in the batch ───────────────────────────
//
// Grid: [batch_size] blocks × 128 threads/block
// Each block processes ONE pre-averaged input vector (p_in[bid, :]) independently.
//
// Parameters:
//   p_in         — [batch_size, d_head] pre-averaged particle embeddings
//   fast_weight  — [d_head, d_head] row-major fast weight matrix W
//   w_gate/w_up  — [d_head, ffn_dim] FFN gate/up projection weights (transposed storage)
//   w_down       — [ffn_dim, d_head] FFN down projection weights
//   p_out        — [batch_size, d_head] output (written in-place for each block)
//   norm_strategy — reserved for future (currently unused; RMSNorm always used)
//   alpha         — subnode's mixing coefficient (Thompson-sampled winner's alpha)
//   sphere_radius — reserved for future energy-sphere constraint (currently unused)
extern "C" __global__ void fused_micro_block_kernel(
    const float* __restrict__ p_in,
    const float* __restrict__ fast_weight,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    float*       __restrict__ p_out,
    int d_head,
    int ffn_dim,
    int norm_strategy,
    float alpha,
    float sphere_radius
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    extern __shared__ float s_mem[];
    float* s_p_in      = s_mem;
    float* s_attn      = s_p_in  + d_head;
    float* s_mid       = s_attn  + d_head;
    float* s_ffn_inter = s_mid   + d_head;
    int max_warps_pad  = ((blockDim.x + WARP_SIZE - 1) / WARP_SIZE + 3) & ~3;
    float* s_reduce    = s_ffn_inter + ffn_dim;
    
    const float* curr_p   = p_in  + bid * d_head;
    float*       curr_out = p_out + bid * d_head;

    // ── Step 0: Load p_in → shared memory and compute RMSNorm scale ──────────
    // Load p_in into s_p_in and simultaneously accumulate sum of squares for RMSNorm.
    // The norm (inv_rms_attn) is used in Step 1 to normalize p_in before the matvec.
    // CPU equivalent: lines 658-662 in micro_block.rs (inference path).
    float sq_sum_attn = 0.0f;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float p_v = __ldg(curr_p + i);
        s_p_in[i] = p_v;
        sq_sum_attn += p_v * p_v;
        s_attn[i] = 0.0f; // Initialize s_attn to zero (will hold matvec output)
    }
    
    sq_sum_attn = block_reduce_sum(sq_sum_attn, s_reduce);
    float inv_rms_attn = rsqrtf(sq_sum_attn / (float)d_head + 1e-8f);
    __syncthreads();

    // ── Step 1: Fast Weight Implicit Memory ───────────────────────────────────
    // attn_out[r] = Σ_c  fast_weight[r, c] * RMSNorm(p_in)[c]
    //             = Σ_c  W[r*d_head + c] * (s_p_in[c] * inv_rms_attn)
    //
    // CPU equivalent (micro_block.rs inference path, lines 664-671):
    //   for r in 0..d_head {
    //     for c in 0..d_head { sum += fast_weight[r*d_head+c] * p_in_normed[c]; }
    //   }
    //
    // GPU implementation: each warp owns a stripe of output rows (wid step).
    // Within each row, lane threads accumulate partial dot products across columns.
    // Results written to s_attn[r] by lane 0 of each warp after intra-warp reduction.
    for (int i = tid; i < d_head; i += blockDim.x) {
        s_attn[i] = 0.0f;
    }
    __syncthreads();

    int lane = tid & 31;
    int wid  = tid >> 5;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    for (int r = wid; r < d_head; r += num_warps) {
        float sum = 0.0f;
        for (int c = lane; c < d_head; c += WARP_SIZE) {
            float normed_in = s_p_in[c] * inv_rms_attn;
            sum += __ldg(fast_weight + r * d_head + c) * normed_in;
        }
        
        // Intra-warp reduction for this row's partial sum
        unsigned int mask = __activemask();
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        
        if (lane == 0) {
            s_attn[r] = sum; // Lane 0 writes the complete row dot product
        }
    }
    __syncthreads();

    // ── Step 2: Additive Residual + Pre-Norm for FFN ──────────────────────────
    // s_mid = p_in + alpha * attn_out
    // Then compute RMSNorm(s_mid) for the FFN input.
    //
    // CPU equivalent (micro_block.rs inference path, lines 678-691):
    //   s_mid[i] = p_in_buf[i] + alpha * attn_out[i]
    //   inv_rms_ffn = 1 / sqrt(mean(s_mid²) + ε)
    float sq_sum_mid = 0.0f;
    for (int i = tid; i < d_head; i += blockDim.x) {
        float val = s_p_in[i] + alpha * s_attn[i];
        s_mid[i] = val;
        sq_sum_mid += val * val;
    }

    sq_sum_mid = block_reduce_sum(sq_sum_mid, s_reduce);
    float inv_rms_ffn = rsqrtf(sq_sum_mid / (float)d_head + 1e-8f);
    __syncthreads();

    // ── Step 3: SwiGLU Activation ─────────────────────────────────────────────
    // For each FFN hidden unit j:
    //   gate_j = Σ_d  RMSNorm(s_mid)[d] * W_gate[d, j]
    //   up_j   = Σ_d  RMSNorm(s_mid)[d] * W_up[d, j]
    //   h_j    = swish(gate_j) * up_j    where swish(x) = x * sigmoid(x)
    //
    // CPU equivalent (micro_block.rs inference path, lines 693-708):
    //   gate += m_val * w_gate[d * ffn_dim + f]  (same [d, f] layout)
    //   up   += m_val * w_up[d * ffn_dim + f]
    //   sig = 1/(1+exp(-gate)); swish = gate*sig; ffn_inter[f] = swish * up
    //
    // Weight layout: w_gate[d * ffn_dim + j] — row = input dim, col = output dim.
    // This is a non-transposed layout (different from standard column-major BLAS).
    // Each thread owns a stripe of j values (outer loop over tid stride).
    for (int j = tid; j < ffn_dim; j += blockDim.x) {
        float gate = 0.0f;
        float up   = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            float m_val = s_mid[d] * inv_rms_ffn; // Apply RMSNorm inline
            gate += m_val * __ldg(w_gate + d * ffn_dim + j);
            up   += m_val * __ldg(w_up   + d * ffn_dim + j);
        }
        float sig   = __fdividef(1.0f, 1.0f + __expf(-gate)); // sigmoid(gate)
        float swish = gate * sig;                               // swish(gate)
        s_ffn_inter[j] = swish * up;                           // SwiGLU output
    }
    __syncthreads();

    // ── Step 4: Down Projection ───────────────────────────────────────────────
    // ffn_out[d] = Σ_j  h_j * W_down[j, d]
    //
    // CPU equivalent (micro_block.rs inference path, lines 705-707):
    //   for (d, t_val) in temp_out.iter_mut().enumerate() {
    //     *t_val += ffn_val * w_down[f * d_head + d];  // w_down[j, d]
    //   }
    //
    // s_attn is reused here to hold the down-projection output (saves shared memory).
    // After Step 1, s_attn is no longer needed — this alias is intentional.
    for (int d = tid; d < d_head; d += blockDim.x) {
        float ffn_acc = 0.0f;
        for (int j = 0; j < ffn_dim; ++j) {
            ffn_acc += s_ffn_inter[j] * __ldg(w_down + j * d_head + d);
        }
        s_attn[d] = ffn_acc; // Reuse s_attn for down-projection output
    }
    __syncthreads();

    // ── Step 5: Additive Residual 2 + Output Clipping ────────────────────────
    // p_out = s_mid + alpha * down_proj_out
    //
    // CPU equivalent (micro_block.rs inference path, lines 710-716):
    //   *out_val = s_val + alpha * temp_val;  (s_val = s_mid, temp_val = down_proj)
    //
    // Note: CPU does NOT apply output clipping. The [-100, 100] clip is a GPU-only
    // numerical guard against NaN propagation from random-initialized weights in
    // the very first forward passes. The clip is wide enough not to affect
    // steady-state operation (expected p_out values are O(1) for unit-norm inputs).
    for (int i = tid; i < d_head; i += blockDim.x) {
        float res = s_mid[i] + alpha * s_attn[i];
        curr_out[i] = fminf(fmaxf(res, -100.0f), 100.0f);
    }
}

// ─── Thread-local buffer pool for async device memory management ──────────────
// Avoids cudaMalloc on every launch. Buffers are grown (never shrunk) using a
// 2x doubling strategy to amortize allocation cost.
struct MicroBlockDeviceBufferPool {
    float* d_p_in       = nullptr;
    float* d_fast_weight = nullptr;
    float* d_w_gate     = nullptr;
    float* d_w_up       = nullptr;
    float* d_w_down     = nullptr;
    float* d_p_out      = nullptr;
    size_t pin_cap = 0;
    size_t kv_cap  = 0;
    size_t w_cap   = 0;
    cudaStream_t last_stream = nullptr;

    ~MicroBlockDeviceBufferPool() {
        if (d_p_in)       { cudaFree(d_p_in);       d_p_in       = nullptr; }
        if (d_fast_weight) { cudaFree(d_fast_weight); d_fast_weight = nullptr; }
        if (d_w_gate)     { cudaFree(d_w_gate);     d_w_gate     = nullptr; }
        if (d_w_up)       { cudaFree(d_w_up);       d_w_up       = nullptr; }
        if (d_w_down)     { cudaFree(d_w_down);     d_w_down     = nullptr; }
        if (d_p_out)      { cudaFree(d_p_out);      d_p_out      = nullptr; }
        cudaGetLastError(); // Clear any error from destructor-time frees
    }

    void ensure_capacity(size_t batch_size, size_t d_head, size_t ffn_dim, cudaStream_t stream) {
        last_stream = stream;

        size_t pin_elems   = batch_size * d_head;
        size_t fw_elems    = d_head * d_head;
        size_t w1_elems    = d_head * ffn_dim;
        size_t w2_elems    = ffn_dim * d_head;
        size_t max_w_elems = w1_elems > w2_elems ? w1_elems : w2_elems;

        if (pin_elems > pin_cap) {
            if (d_p_in)  cudaFreeAsync(d_p_in,  stream);
            if (d_p_out) cudaFreeAsync(d_p_out, stream);
            pin_cap = pin_elems * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_p_in,  pin_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_p_out, pin_cap * sizeof(float), stream));
        }
        if (fw_elems > kv_cap) {
            if (d_fast_weight) cudaFreeAsync(d_fast_weight, stream);
            kv_cap = fw_elems * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_fast_weight, kv_cap * sizeof(float), stream));
        }
        if (max_w_elems > w_cap) {
            if (d_w_gate) cudaFreeAsync(d_w_gate, stream);
            if (d_w_up)   cudaFreeAsync(d_w_up,   stream);
            if (d_w_down) cudaFreeAsync(d_w_down, stream);
            w_cap = max_w_elems * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_w_gate, w_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_w_up,   w_cap * sizeof(float), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_w_down, w_cap * sizeof(float), stream));
        }
    }
};

// One pool per thread (MicroBlockNode processes its own particles independently).
static thread_local MicroBlockDeviceBufferPool g_micro_pool;

// ─── C FFI Entry Point ────────────────────────────────────────────────────────
// Called from annp-cuda/src/ffi.rs via extern "C".
//
// Launches fused_micro_block_kernel with:
//   - 128 threads/block (good occupancy for d_head=64; warp-efficient)
//   - 1 block/particle (each particle independent, no cross-particle synchronization)
//   - Shared memory sized for 3*d_head + ffn_dim + max_warps_pad floats
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
    const float* dev_pin         = get_device_ptr(p_in,        g_micro_pool.d_p_in,       batch_size * d_head, stream, &was_host_copied);
    const float* dev_fast_weight = get_device_ptr(fast_weight, g_micro_pool.d_fast_weight, d_head * d_head,    stream, &was_host_copied);
    const float* dev_wgate       = get_device_ptr(w_gate,      g_micro_pool.d_w_gate,      d_head * ffn_dim,   stream, &was_host_copied);
    const float* dev_wup         = get_device_ptr(w_up,        g_micro_pool.d_w_up,        d_head * ffn_dim,   stream, &was_host_copied);
    const float* dev_wdown       = get_device_ptr(w_down,      g_micro_pool.d_w_down,      ffn_dim * d_head,   stream, &was_host_copied);

    float* dev_pout = get_device_ptr_mut(p_out, g_micro_pool.d_p_out, batch_size * d_head, stream, &was_host_copied);

    int threads_per_block = 128;
    int blocks      = batch_size; // 1 block per particle
    int dh_pad      = (d_head + 3) & ~3;
    int ffn_pad     = (ffn_dim + 3) & ~3;
    int max_warps     = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    int max_warps_pad = (max_warps + 3) & ~3;

    // Shared memory: s_p_in(dh) + s_attn(dh) + s_mid(dh) + s_ffn_inter(ffn) + s_reduce(warps)
    size_t shared_mem_bytes = (3 * dh_pad + ffn_pad + max_warps_pad) * sizeof(float);

    fused_micro_block_kernel<<<blocks, threads_per_block, shared_mem_bytes, stream>>>(
        dev_pin, dev_fast_weight, dev_wgate, dev_wup, dev_wdown, dev_pout,
        d_head, ffn_dim, norm_strategy, alpha, sphere_radius
    );

    // Copy result back to host if p_out is a host pointer (with stream sync)
    copy_back_if_host(p_out, dev_pout, batch_size * d_head, stream);
}
