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

struct AggregatorDeviceBufferPool {
    float* d_src = nullptr;
    float* d_dst = nullptr;
    int* d_indices = nullptr;
    size_t cap = 0;

    void ensure_capacity(size_t num_particles, size_t d_head) {
        if (num_particles > cap) {
            if (d_src) cudaFree(d_src);
            if (d_dst) cudaFree(d_dst);
            if (d_indices) cudaFree(d_indices);

            cap = num_particles * 2 + 512;
            cudaMalloc(&d_src, cap * d_head * sizeof(float));
            cudaMalloc(&d_dst, cap * d_head * sizeof(float));
            cudaMalloc(&d_indices, cap * sizeof(int));
        }
    }
};

static thread_local AggregatorDeviceBufferPool g_agg_pool;

// C-FFI Wrapper Function
extern "C" void launch_particle_prefetch_aggregate(
    const float* src_particles,
    float* dst_buffer,
    const int* active_indices,
    int num_particles,
    int d_head,
    cudaStream_t stream
) {
    if (num_particles <= 0 || d_head <= 0) return;

    g_agg_pool.ensure_capacity(num_particles, d_head);

    if (src_particles) cudaMemcpyAsync(g_agg_pool.d_src, src_particles, num_particles * d_head * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (active_indices) cudaMemcpyAsync(g_agg_pool.d_indices, active_indices, num_particles * sizeof(int), cudaMemcpyHostToDevice, stream);

    int threads_per_block = 128;
    int blocks = (num_particles + threads_per_block - 1) / threads_per_block;

    particle_prefetch_aggregate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        g_agg_pool.d_src, g_agg_pool.d_dst, g_agg_pool.d_indices, num_particles, d_head
    );

    if (dst_buffer) cudaMemcpyAsync(dst_buffer, g_agg_pool.d_dst, num_particles * d_head * sizeof(float), cudaMemcpyDeviceToHost, stream);
}
