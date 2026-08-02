// annp-cuda/cuda/common.cuh
//
// Shared CUDA utilities: struct layout, warp primitives, and memory helpers.
//
// ALIGNMENT CONTRACT:
//   ParticleCudaHeader must match the Rust struct annp_core::ParticleCudaHeader
//   byte-for-byte. Changes here require a matching change in annp-cuda/src/ffi.rs.
//   The static_assert below enforces this at compile time.
//
// MEMORY MANAGEMENT STRATEGY:
//   All device buffers use thread_local pool allocators (MicroBlockDeviceBufferPool,
//   RouterDeviceBufferPool). Buffers are allocated with cudaMallocAsync and grown
//   only when needed (lazy doubling). This avoids synchronous cudaMalloc on the
//   hot path while remaining Nsight Profiler compatible.
//
//   get_device_ptr / get_device_ptr_mut: zero-copy fast path for data already on
//   the device (e.g., managed memory). Falls back to H2D copy otherwise.
//   copy_back_if_host: D2H copy + stream sync only when the output pointer is a
//   host buffer. If it's already a device pointer, skip the copy.

#ifndef ANNP_CUDA_COMMON_CUH
#define ANNP_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffff

// Non-fatal CUDA error logger.
// WHY NON-FATAL? Dynamic buffer management (MallocAsync → Malloc fallback) means
// some calls may fail gracefully. Aborting on every error would break the fallback
// path. Errors that matter will surface as incorrect results or subsequent failures.
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* Non-fatal: caller continues with degraded functionality */ \
        } \
    } while (0)

// cudaMallocAsync with synchronous fallback.
//
// cudaMallocAsync is preferred (stream-ordered, compatible with Nsight profiler and
// memory pools). If it fails (e.g., stream pool unavailable on older drivers), we fall
// back to synchronous cudaMalloc. The sticky error state from the failed async call is
// cleared before the fallback to prevent cascading errors downstream.
template <typename T>
inline cudaError_t safe_cuda_malloc_async(T** ptr, size_t size, cudaStream_t stream) {
    if (!ptr || size == 0) return cudaSuccess;
    cudaError_t err = cudaMallocAsync((void**)ptr, size, stream);
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear sticky error before fallback
        err = cudaMalloc((void**)ptr, size);
    }
    return err;
}

// SwiGLU activation: swish(gate) * up = (gate * sigmoid(gate)) * up.
//
// This matches the CPU path in micro_block.rs:
//   let sig = 1.0 / (1.0 + (-gate).exp());
//   let swish = gate * sig;
//   ffn_inter[f] = swish * up;
//
// NOTE: This helper is defined but not used by the fused kernel (which inlines
// the computation for better register pressure control). Kept for reference.
__device__ inline float swiglu(float gate, float up) {
    float swish = gate / (1.0f + __expf(-gate));
    return swish * up;
}

// ParticleCudaHeader — must match annp_core::ParticleCudaHeader exactly (16 bytes).
//
// Layout rationale:
//   - origin_token_id (u32, 4B): global monotone token position for TD learning (dt = t - t_prev).
//   - shard_id (u16, 2B): which embedding shard this particle carries.
//   - pad0 (2B): alignment padding before float energy (float requires 4B alignment).
//   - energy (f32, 4B): remaining particle energy; reaches 0 at max_hop via linear decay.
//   - hop_count (u16, 2B): number of hops taken so far; guards min_hop halt condition.
//   - halted (u8, 1B): 0 = active, 1 = halted. u8 chosen to minimize struct size.
//   - pad1 (1B): trailing alignment to reach 16 bytes (cache-line subdivision).
//
// The static_assert below will cause a compile error if Rust changes its layout
// (e.g., reorders fields) without updating this header.
struct ParticleCudaHeader {
    uint32_t origin_token_id; // 0..4
    uint16_t shard_id;        // 4..6
    uint8_t pad0[2];          // 6..8  explicit padding before float energy
    float energy;             // 8..12
    uint16_t hop_count;       // 12..14
    uint8_t halted;           // 14..15
    uint8_t pad1[1];          // 15..16 trailing padding
};

static_assert(sizeof(ParticleCudaHeader) == 16, "ParticleCudaHeader size must be exactly 16 bytes");

// ─── Warp Reduction Primitives ────────────────────────────────────────────────
//
// __shfl_down_sync with __activemask() instead of FULL_WARP_MASK:
// In partial warps (blockDim.x not a multiple of 32), using FULL_WARP_MASK on
// inactive lanes is undefined behavior. __activemask() limits shuffle to the
// set of currently active threads, which is safe for all block sizes.

// Sum reduction across a warp: returns the sum in lane 0 (and all lanes after
// the loop, but only lane 0's value is meaningful for inter-warp coordination).
__device__ inline float warp_reduce_sum(float val) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Max reduction across a warp.
__device__ inline float warp_reduce_max(float val) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

// Integer max reduction across a warp (used for argmax index propagation).
__device__ inline int warp_reduce_max_int(int val) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

// ─── Host/Device Pointer Helpers ──────────────────────────────────────────────
//
// Zero-copy fast path: if the input pointer is already a CUDA device or managed
// pointer, return it directly without memcpy. Otherwise, copy to the pre-allocated
// pool buffer and return that.
//
// WHY cudaPointerGetAttributes on every call?
// ANNP Rust code passes raw host pointers (Vec<f32>) at training time, and may
// pass device pointers at inference time (if CUDA buffers are reused). Rather
// than requiring the caller to tag every pointer, we probe the CUDA runtime.
// The overhead is one CUDA API call per launch, which is negligible compared to
// the kernel execution time.
//
// cudaGetLastError() clears the sticky error set by cudaPointerGetAttributes when
// given a pure host pointer (which is expected; it's not an error for us).

template <typename T>
inline const T* get_device_ptr(const T* host_or_dev, T* d_pool_buf, size_t count, cudaStream_t stream, bool* was_host_copied = nullptr) {
    if (!host_or_dev || count == 0) return d_pool_buf;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, host_or_dev);
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)) {
        return host_or_dev; // Zero-copy: already on device
    }
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear sticky error from host pointer probe
    }
    if (d_pool_buf) {
        cudaMemcpyAsync(d_pool_buf, host_or_dev, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        if (was_host_copied) *was_host_copied = true;
    }
    return d_pool_buf;
}

template <typename T>
inline T* get_device_ptr_mut(T* host_or_dev, T* d_pool_buf, size_t count, cudaStream_t stream, bool* was_host_copied = nullptr) {
    if (!host_or_dev || count == 0) return d_pool_buf;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, host_or_dev);
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)) {
        return host_or_dev; // Zero-copy: already on device
    }
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear sticky error
    }
    if (d_pool_buf) {
        if (was_host_copied) *was_host_copied = true;
    }
    return d_pool_buf;
}

// Copy device buffer back to host if the destination is a host pointer.
//
// WHY cudaStreamSynchronize on H2D copy?
// Rust's Vec<T> is non-pinned host memory. cudaMemcpyAsync to non-pinned host
// memory is technically allowed but the transfer is NOT guaranteed to be visible
// until the stream completes. Without synchronization, Rust may read stale data.
// Synchronizing here is the safest option; for high-throughput workloads, the
// caller can switch to pinned host memory (cudaHostAlloc) to eliminate the sync.
template <typename T>
inline bool copy_back_if_host(T* host_or_dev, const T* d_buf, size_t count, cudaStream_t stream) {
    if (!host_or_dev || !d_buf || count == 0 || (const void*)host_or_dev == (const void*)d_buf) return false;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, host_or_dev);
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)) {
        if ((const void*)host_or_dev != (const void*)d_buf) {
            cudaMemcpyAsync(host_or_dev, d_buf, count * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
        return false; // No host sync needed for D2D copy
    } else {
        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear sticky error
        }
        cudaMemcpyAsync(host_or_dev, d_buf, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        // Synchronize: Rust reads this buffer immediately after launch_*() returns.
        // Without sync, Rust would observe uninitialized / stale values.
        cudaStreamSynchronize(stream);
        return true;
    }
}

#endif // ANNP_CUDA_COMMON_CUH
