#include "common.cuh"

/**
 * CUDA Kernel for Particle Q-Routing & Halting Evaluation.
 */
extern "C" __global__ void particle_router_kernel(
    const float* __restrict__ p_in,          // [batch_size, d_head]
    const float* __restrict__ p_out,         // [batch_size, d_head]
    const float* __restrict__ routing_table, // [d_head, num_neighbors]
    const float* __restrict__ gumbel_noise,  // [batch_size, num_neighbors]
    int* __restrict__ chosen_neighbor,       // [batch_size]
    bool* __restrict__ halting_flags,        // [batch_size]
    int batch_size,
    int d_head,
    int num_neighbors,
    float temperature,
    float epsilon_p,
    float epsilon_h,
    int min_hop,
    const ParticleCudaHeader* __restrict__ headers
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= batch_size) return;

    const float* pin = p_in + pid * d_head;
    const float* pout = p_out + pid * d_head;
    const float* gnoise = gumbel_noise + pid * num_neighbors;

    // 1. Evaluate Delta p: ||p_out - p_in||_2
    float delta_p_sq = 0.0f;
    for (int d = 0; d < d_head; ++d) {
        float diff = pout[d] - pin[d];
        delta_p_sq += diff * diff;
    }
    float delta_p = sqrtf(delta_p_sq);

    // 2. Evaluate Routing logits: p_out * Routing_Table
    float logits[16]; // Max num_neighbors supported per thread = 16
    float max_logit = -1e9f;

    for (int k = 0; k < num_neighbors; ++k) {
        float dot = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            dot += pout[d] * routing_table[d * num_neighbors + k];
        }
        logits[k] = dot;
        if (dot > max_logit) max_logit = dot;
    }

    // Compute Softmax probabilities and Shannon Entropy H
    float sum_exp = 0.0f;
    float probs[16];
    for (int k = 0; k < num_neighbors; ++k) {
        float exp_val = expf((logits[k] - max_logit) / temperature);
        probs[k] = exp_val;
        sum_exp += exp_val;
    }

    float entropy = 0.0f;
    for (int k = 0; k < num_neighbors; ++k) {
        probs[k] /= sum_exp;
        if (probs[k] > 1e-10f) {
            entropy -= probs[k] * log2f(probs[k] + 1e-10f);
        }
    }

    // 3. Double convergence Halting check
    ParticleCudaHeader header = headers[pid];
    bool is_halted = header.halted;

    if (!is_halted && header.hop_count >= min_hop) {
        if (delta_p < epsilon_p && entropy < epsilon_h) {
            is_halted = true;
        }
    }
    halting_flags[pid] = is_halted;

    // 4. Next-hop neighbor selection via Gumbel-Softmax discrete argmax
    int best_neighbor = 0;
    float max_gumbel_val = -1e9f;
    for (int k = 0; k < num_neighbors; ++k) {
        float g_val = logf(probs[k] + 1e-10f) + gnoise[k];
        if (g_val > max_gumbel_val) {
            max_gumbel_val = g_val;
            best_neighbor = k;
        }
    }
    chosen_neighbor[pid] = best_neighbor;
}

// C-FFI Wrapper Function
extern "C" void launch_particle_router(
    const float* p_in,
    const float* p_out,
    const float* routing_table,
    const float* gumbel_noise,
    int* chosen_neighbor,
    bool* halting_flags,
    int batch_size,
    int d_head,
    int num_neighbors,
    float temperature,
    float epsilon_p,
    float epsilon_h,
    int min_hop,
    const ParticleCudaHeader* headers,
    cudaStream_t stream
) {
    int threads_per_block = 64;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    particle_router_kernel<<<blocks, threads_per_block, 0, stream>>>(
        p_in, p_out, routing_table, gumbel_noise, chosen_neighbor, halting_flags,
        batch_size, d_head, num_neighbors, temperature, epsilon_p, epsilon_h, min_hop, headers
    );
}
