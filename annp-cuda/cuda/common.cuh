#ifndef ANNP_CUDA_COMMON_CUH
#define ANNP_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

#define MAX_D_HEAD 128
#define WARP_SIZE 32

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

#endif // ANNP_CUDA_COMMON_CUH
