#include "common.cuh"

/**
 * Industrial-Grade Parallel Particle Aggregator Kernel.
 * 1. Coalesced Global Memory Access & Vectorized float4 Loading.
 * 2. GPU Dynamic Max Index Reduction for Safe Array Bounds.
 * 3. Race-Free Stream-Ordered Memory Management.
 */

extern "C" __global__ void find_max_index_kernel(
    const int* __restrict__ active_indices,
    int num_particles,
    int* __restrict__ d_max_idx
) {
    int tid = threadIdx.x;
    int local_max = (num_particles > 0 && active_indices) ? active_indices[0] : 0;

    for (int i = tid; i < num_particles; i += blockDim.x) {
        int idx = active_indices[i];
        if (idx > local_max) {
            local_max = idx;
        }
    }

    int w_max = warp_reduce_max_int(local_max);
    if (tid == 0) {
        *d_max_idx = w_max;
    }
}

extern "C" __global__ void particle_prefetch_aggregate_kernel(
    const float* __restrict__ src_particles, // [max_idx + 1, d_head]
    float* __restrict__ dst_buffer,           // [num_particles, d_head]
    const int* __restrict__ active_indices,   // [num_particles] or nullptr
    int num_particles,
    int d_head
) {
    int pid = blockIdx.x;
    if (pid >= num_particles) return;

    int tid = threadIdx.x;
    int src_idx = (active_indices != nullptr) ? active_indices[pid] : pid;

    const float* src = src_particles + src_idx * d_head;
    float* dst = dst_buffer + pid * d_head;

    bool aligned = (((uintptr_t)src | (uintptr_t)dst) & 15) == 0 && (d_head % 4 == 0);

    if (aligned) {
        int d_head4 = d_head / 4;
        const float4* src4 = reinterpret_cast<const float4*>(src);
        float4* dst4 = reinterpret_cast<float4*>(dst);

        for (int i = tid; i < d_head4; i += blockDim.x) {
            dst4[i] = __ldg(src4 + i);
        }
    } else {
        for (int i = tid; i < d_head; i += blockDim.x) {
            dst[i] = __ldg(src + i);
        }
    }
}

struct AggregatorDeviceBufferPool {
    float* d_src = nullptr;
    float* d_dst = nullptr;
    int* d_indices = nullptr;
    int* d_max_idx = nullptr;
    size_t src_cap = 0;
    size_t dst_cap = 0;
    size_t idx_cap = 0;
    cudaStream_t last_stream = nullptr;

    ~AggregatorDeviceBufferPool() {
        if (d_src) { cudaFree(d_src); d_src = nullptr; }
        if (d_dst) { cudaFree(d_dst); d_dst = nullptr; }
        if (d_indices) { cudaFree(d_indices); d_indices = nullptr; }
        if (d_max_idx) { cudaFree(d_max_idx); d_max_idx = nullptr; }
        cudaGetLastError();
    }

    void ensure_capacity(size_t src_floats, size_t dst_floats, size_t num_particles, cudaStream_t stream) {
        last_stream = stream;

        if (src_floats > src_cap) {
            if (d_src) cudaFreeAsync(d_src, stream);
            src_cap = src_floats * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_src, src_cap * sizeof(float), stream));
        }

        if (dst_floats > dst_cap) {
            if (d_dst) cudaFreeAsync(d_dst, stream);
            dst_cap = dst_floats * 2 + 1024;
            CHECK_CUDA(safe_cuda_malloc_async(&d_dst, dst_cap * sizeof(float), stream));
        }

        if (num_particles > idx_cap) {
            if (d_indices) cudaFreeAsync(d_indices, stream);
            if (d_max_idx) cudaFreeAsync(d_max_idx, stream);
            idx_cap = num_particles * 2 + 512;
            CHECK_CUDA(safe_cuda_malloc_async(&d_indices, idx_cap * sizeof(int), stream));
            CHECK_CUDA(safe_cuda_malloc_async(&d_max_idx, sizeof(int), stream));
        }
    }
};

static thread_local AggregatorDeviceBufferPool g_agg_pool;

// C-FFI Wrapper Function for Prefetch & Aggregation
extern "C" void launch_particle_prefetch_aggregate(
    const float* src_particles,
    float* dst_buffer,
    const int* active_indices,
    int num_particles,
    int d_head,
    cudaStream_t stream
) {
    if (num_particles <= 0 || d_head <= 0 || !src_particles || !dst_buffer) return;

    int max_idx = num_particles - 1;

    if (active_indices != nullptr) {
        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, active_indices);
        bool is_idx_device = (err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged));
        if (err != cudaSuccess) {
            cudaGetLastError();
        }

        if (!is_idx_device) {
            for (int i = 0; i < num_particles; ++i) {
                if (active_indices[i] > max_idx) {
                    max_idx = active_indices[i];
                }
            }
        } else {
            // Active indices are on GPU.
            // Find max index using mini GPU kernel to safely size array bounds.
            g_agg_pool.ensure_capacity(0, 0, num_particles, stream);
            find_max_index_kernel<<<1, 32, 0, stream>>>(active_indices, num_particles, g_agg_pool.d_max_idx);
            cudaMemcpyAsync(&max_idx, g_agg_pool.d_max_idx, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }

    if (max_idx < num_particles - 1) {
        max_idx = num_particles - 1;
    }

    size_t total_src_floats = (size_t)(max_idx + 1) * d_head;
    size_t total_dst_floats = (size_t)num_particles * d_head;

    g_agg_pool.ensure_capacity(total_src_floats, total_dst_floats, num_particles, stream);

    bool was_host_copied = false;
    const float* dev_src = get_device_ptr(src_particles, g_agg_pool.d_src, total_src_floats, stream, &was_host_copied);
    const int* dev_idx = (active_indices != nullptr) ? get_device_ptr(active_indices, g_agg_pool.d_indices, num_particles, stream, &was_host_copied) : nullptr;
    float* dev_dst = get_device_ptr_mut(dst_buffer, g_agg_pool.d_dst, total_dst_floats, stream, &was_host_copied);

    int threads_per_block = 128;
    int blocks = num_particles; // 1 block per particle

    particle_prefetch_aggregate_kernel<<<blocks, threads_per_block, 0, stream>>>(
        dev_src, dev_dst, dev_idx, num_particles, d_head
    );

    copy_back_if_host(dst_buffer, dev_dst, total_dst_floats, stream);
}
