#ifndef ANNP_CUDA_COMMON_CUH
#define ANNP_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

#define MAX_D_HEAD 128
#define WARP_SIZE 32
#define FULL_WARP_MASK 0xffffffff

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while (0)

struct ParticleCudaHeader {
    unsigned int origin_token_id;
    unsigned short shard_id;
    float energy;
    unsigned short hop_count;
    bool halted;
};

// Hardware Fast Swish activation: x * sigmoid(x)
__device__ __forceinline__ float swish(float x) {
    return x / (1.0f + __expf(-x));
}

// Hardware Fast SwiGLU activation for FFN
__device__ __forceinline__ float swiglu(float gate, float up) {
    return swish(gate) * up;
}

// Full 32-thread Warp-level parallel sum reduction using registers (__shfl_down_sync)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

// Full 32-thread Warp-level parallel max reduction using registers (__shfl_down_sync)
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(FULL_WARP_MASK, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Vectorized 128-bit float4 copy helper (16 bytes per instruction) with alignment safety
__device__ __forceinline__ void copy_float4(float* dst, const float* src, int count) {
    bool aligned = (((uintptr_t)dst | (uintptr_t)src) & 15) == 0;
    if (aligned) {
        int num_float4 = count / 4;
        float4* d = reinterpret_cast<float4*>(dst);
        const float4* s = reinterpret_cast<const float4*>(src);
        
        for (int i = 0; i < num_float4; ++i) {
            d[i] = s[i];
        }
        
        int remainder = count % 4;
        int start_idx = count - remainder;
        for (int i = 0; i < remainder; ++i) {
            dst[start_idx + i] = src[start_idx + i];
        }
    } else {
        for (int i = 0; i < count; ++i) {
            dst[i] = src[i];
        }
    }
}

// Host/Device memory detection & Zero-Copy helper
template <typename T>
inline const T* get_device_ptr(const T* host_or_dev, T* d_pool_buf, size_t count, cudaStream_t stream) {
    if (!host_or_dev || count == 0) return d_pool_buf;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, host_or_dev) == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
        return host_or_dev;
    }
    if (d_pool_buf) {
        cudaMemcpyAsync(d_pool_buf, host_or_dev, count * sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    return d_pool_buf;
}

template <typename T>
inline T* get_device_ptr_mut(T* host_or_dev, T* d_pool_buf, size_t count, cudaStream_t stream) {
    if (!host_or_dev || count == 0) return d_pool_buf;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, host_or_dev) == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
        return host_or_dev;
    }
    return d_pool_buf;
}

template <typename T>
inline void copy_back_if_host(T* host_or_dev, const T* d_buf, size_t count, cudaStream_t stream) {
    if (!host_or_dev || !d_buf || count == 0 || (const void*)host_or_dev == (const void*)d_buf) return;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, host_or_dev) == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
        if ((const void*)host_or_dev != (const void*)d_buf) {
            cudaMemcpyAsync(host_or_dev, d_buf, count * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
    } else {
        cudaMemcpyAsync(host_or_dev, d_buf, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }
}

#endif // ANNP_CUDA_COMMON_CUH
