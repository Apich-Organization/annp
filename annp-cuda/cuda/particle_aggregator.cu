#include "common.cuh"

/**
 * Industrial-Standard Vectorized CUDA Aggregator & Prefetching Ring Buffer Kernel.
 * Uses 128-bit float4 instructions to maximize memory bandwidth during particle prefetching and aggregation.
 */
extern "C" __global__ void particle_prefetch_aggregate_kernel(
    const float* __restrict__ src_particles,  // Scattered input particles [N, d_head]
    float* __restrict__ dst_buffer,            // Contiguous prefetched output buffer [N, d_head]
    const int* __restrict__ active_indices,     // Indices of active particles to prefetch
    int num_particles,
    int d_head
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    int src_idx = active_indices[idx];
    const float* src_ptr = src_particles + src_idx * d_head;
    float* dst_ptr = dst_buffer + idx * d_head;

    // Vectorized 128-bit memory copy (16 bytes per float4 instruction)
    copy_float4(dst_ptr, src_ptr, d_head);
}

// C-FFI Wrapper Function
extern "C" void launch_particle_prefetch_aggregate(
    const float* src_particles,
    float* dst_buffer,
    const int* active_indices,
    int num_particles,
    int d_head,
    cudaStream_t stream
) {
    int threads_per_block = 128;
    int blocks = (num_particles + threads_per_block - 1) / threads_per_block;

    particle_prefetch_aggregate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        src_particles, dst_buffer, active_indices, num_particles, d_head
    );
}
