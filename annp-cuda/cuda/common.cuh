#ifndef ANNP_CUDA_COMMON_CUH
#define ANNP_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffff

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* Non-fatal logging for dynamic memory management */ \
        } \
    } while (0)

// Helper: Safe dynamic allocation fallback if cudaMallocAsync is unsupported
inline cudaError_t safe_cuda_malloc_async(void** ptr, size_t size, cudaStream_t stream) {
    cudaError_t err = cudaMallocAsync(ptr, size, stream);
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear error state
        err = cudaMalloc(ptr, size);
    }
    return err;
}

// SwiGLU activation helper
__device__ inline float swiglu(float gate, float up) {
    float swish = gate / (1.0f + __expf(-gate));
    return swish * up;
}

// Struct layout matching Rust ParticleCudaHeader exactly (16 bytes)
struct ParticleCudaHeader {
    uint32_t origin_token_id; // 0..4
    uint16_t shard_id;        // 4..6
    uint8_t pad0[2];          // 6..8 explicit padding before float energy
    float energy;             // 8..12
    uint16_t hop_count;       // 12..14
    uint8_t halted;           // 14..15
    uint8_t pad1[1];          // 15..16 trailing padding
};

static_assert(sizeof(ParticleCudaHeader) == 16, "ParticleCudaHeader size must be exactly 16 bytes");

// Warp reduction primitives
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

__device__ inline int warp_reduce_max_int(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

// Host/Device memory detection & Zero-Copy helper with Sticky Error Clearing
template <typename T>
inline const T* get_device_ptr(const T* host_or_dev, T* d_pool_buf, size_t count, cudaStream_t stream, bool* was_host_copied = nullptr) {
    if (!host_or_dev || count == 0) return d_pool_buf;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, host_or_dev);
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)) {
        return host_or_dev;
    }
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear sticky CUDA error status from invalid host pointer inspection
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
        return host_or_dev;
    }
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear sticky CUDA error status
    }
    if (d_pool_buf) {
        if (was_host_copied) *was_host_copied = true;
    }
    return d_pool_buf;
}

template <typename T>
inline bool copy_back_if_host(T* host_or_dev, const T* d_buf, size_t count, cudaStream_t stream) {
    if (!host_or_dev || !d_buf || count == 0 || (const void*)host_or_dev == (const void*)d_buf) return false;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, host_or_dev);
    if (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged)) {
        if ((const void*)host_or_dev != (const void*)d_buf) {
            cudaMemcpyAsync(host_or_dev, d_buf, count * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
        return false;
    } else {
        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear sticky CUDA error status
        }
        cudaMemcpyAsync(host_or_dev, d_buf, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        return true;
    }
}

#endif // ANNP_CUDA_COMMON_CUH
