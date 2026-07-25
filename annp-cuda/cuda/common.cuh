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

// Swish activation: x * sigmoid(x)
__device__ __forceinline__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

// SwiGLU activation for FFN
__device__ __forceinline__ float swiglu(float gate, float up) {
    return swish(gate) * up;
}

// Warp-level parallel sum reduction using registers (shuffle)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

// Warp-level parallel max reduction using registers (shuffle)
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(FULL_WARP_MASK, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Vectorized 128-bit float4 copy helper (16 bytes per instruction)
__device__ __forceinline__ void copy_float4(float* dst, const float* src, int count) {
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
}

#endif // ANNP_CUDA_COMMON_CUH
