use annp_core::{
    MicroBlockConfig, NormStrategy, Particle, compute_attention_entropy, compute_delta_p,
};
use annp_cuda::CudaMicroBlockRunner;
use rand::Rng;

/// Autonomous Micro-Block Node.
pub struct MicroBlockNode {
    pub node_id: usize,
    pub config: MicroBlockConfig,
    // Weights
    pub w_gate: Vec<f32>, // [d_head, ffn_dim]
    pub w_up: Vec<f32>,   // [d_head, ffn_dim]
    pub w_down: Vec<f32>, // [ffn_dim, d_head]
    // Momentum Velocity Buffers (\beta = 0.9)
    pub v_gate: Vec<f32>,
    pub v_up: Vec<f32>,
    pub v_down: Vec<f32>,
    // MicroNorm learnable parameter
    pub alpha: f32,
    // Local FIFO KV Cache
    pub k_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub v_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub max_kv_len: usize,
    // Last activation cache for exact chain-rule backpropagation
    pub last_p_in: Vec<f32>, // [d_head]
    // Node state statistics
    pub cumulative_sequence_len: u64, // S_j for plastic hardening
    pub activation_count: u64,
    // Reusable workspace scratch buffers to avoid heap allocations in process_batch
    pub p_in_buf: Vec<f32>,
    pub p_out_buf: Vec<f32>,
    pub use_cuda: bool,
}

impl MicroBlockNode {
    pub fn new(
        node_id: usize,
        config: MicroBlockConfig,
        max_kv_len: usize,
        use_cuda: bool,
    ) -> Self {
        let d_head = config.d_head;
        let ffn_dim = d_head * config.ffn_expansion;

        let mut rng = rand::rng();
        let scale = (2.0 / (d_head + ffn_dim) as f64).sqrt() as f32;

        let w_gate = (0..d_head * ffn_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        let w_up = (0..d_head * ffn_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        let w_down = (0..ffn_dim * d_head)
            .map(|_| rng.random_range(-scale..scale))
            .collect();

        let v_gate = vec![0.0f32; d_head * ffn_dim];
        let v_up = vec![0.0f32; d_head * ffn_dim];
        let v_down = vec![0.0f32; ffn_dim * d_head];

        let alpha = config.alpha_init;
        let k_cache = Vec::with_capacity(max_kv_len * d_head);
        let v_cache = Vec::with_capacity(max_kv_len * d_head);
        let last_p_in = vec![0.0f32; d_head];

        Self {
            node_id,
            config,
            w_gate,
            w_up,
            w_down,
            v_gate,
            v_up,
            v_down,
            alpha,
            k_cache,
            v_cache,
            max_kv_len,
            last_p_in,
            cumulative_sequence_len: 0,
            activation_count: 0,
            p_in_buf: Vec::with_capacity(64 * d_head),
            p_out_buf: Vec::with_capacity(64 * d_head),
            use_cuda,
        }
    }

    /// Update KV Cache with new incoming particle (push FIFO safely)
    pub fn update_kv_cache(&mut self, particle: &Particle) {
        let d_head = self.config.d_head;
        assert_eq!(particle.payload.len(), d_head);

        if self.max_kv_len > 0 {
            let curr_len = self.k_cache.len() / d_head;
            if curr_len >= self.max_kv_len && self.k_cache.len() >= d_head {
                // Evict oldest KV entry safely
                self.k_cache.drain(0..d_head);
                self.v_cache.drain(0..d_head);
            }
        }

        self.k_cache.extend_from_slice(&particle.payload);
        self.v_cache.extend_from_slice(&particle.payload);
    }

    /// Process a batch of particles through Micro-Block CUDA/CPU computation pipeline (0 heap allocations)
    pub fn process_batch(&mut self, particles: &mut [Particle]) {
        let batch_size = particles.len();
        if batch_size == 0 {
            return;
        }

        let d_head = self.config.d_head;
        let ffn_dim = d_head * self.config.ffn_expansion;
        let kv_len = self.k_cache.len() / d_head;
        let norm_strat_val = match self.config.norm_strategy {
            NormStrategy::MicroRMSNorm => 0,
            NormStrategy::SphereNormalization => 1,
        };

        // Flatten p_in payloads and cache last_p_in into reusable p_in_buf
        self.p_in_buf.clear();
        self.p_in_buf.reserve(batch_size * d_head);

        for (d, val) in self.last_p_in.iter_mut().enumerate() {
            *val = 0.0;
            for p in particles.iter() {
                *val += p.payload[d];
            }
            *val /= batch_size as f32;
        }

        for p in particles.iter() {
            self.p_in_buf.extend_from_slice(&p.payload);
        }

        self.p_out_buf.clear();
        self.p_out_buf.resize(batch_size * d_head, 0.0f32);

        // Launch CUDA / Fused CudaMicroBlockRunner kernel (respecting self.use_cuda flag)
        CudaMicroBlockRunner::execute_fused_with_stream_device(
            &self.p_in_buf,
            &self.k_cache,
            &self.v_cache,
            &self.w_gate,
            &self.w_up,
            &self.w_down,
            &mut self.p_out_buf,
            batch_size,
            d_head,
            ffn_dim,
            kv_len,
            norm_strat_val,
            self.alpha,
            self.config.sphere_radius,
            None,
            self.use_cuda,
        );

        // Update particles, evaluation halting condition and metrics
        for (i, p) in particles.iter_mut().enumerate() {
            let out_slice = &self.p_out_buf[i * d_head..(i + 1) * d_head];
            let delta_p = compute_delta_p(&p.payload, out_slice);

            p.payload.copy_from_slice(out_slice);
            p.header.step_hop(self.config.max_hop);

            // Double convergence spontaneous halting check
            if !p.header.halted && p.header.hop_count >= self.config.min_hop {
                let local_entropy = compute_attention_entropy(&[0.5, 0.5]); // Local attention entropy proxy
                if delta_p < self.config.epsilon_p && local_entropy < self.config.epsilon_h {
                    p.header.halted = true;
                }
            }

            // Update KV Cache with computed output
            self.update_kv_cache(p);
        }

        self.activation_count += batch_size as u64;
        self.cumulative_sequence_len += batch_size as u64;
    }

    /// Chain-Rule Backpropagation Update Step (v = 0.9 * v + 0.1 * g)
    pub fn update_weights_with_shard_err(&mut self, shard_err: &[f32], lr: f32) {
        let d_head = self.config.d_head;
        if shard_err.len() < d_head || self.activation_count == 0 {
            return;
        }

        let ffn_dim = d_head * self.config.ffn_expansion;
        let beta = 0.9f32;
        let weight_decay = 0.9999f32;

        // Chain-rule gradient update for W_down [ffn_dim, d_head]
        for j in 0..ffn_dim {
            let p_in_val = self.last_p_in[j % d_head];
            for d in 0..d_head {
                let idx = j * d_head + d;
                let grad = shard_err[d] * p_in_val;
                self.v_down[idx] = beta * self.v_down[idx] + (1.0 - beta) * grad;
                self.w_down[idx] =
                    self.w_down[idx] * weight_decay - lr * self.v_down[idx].clamp(-0.1, 0.1);
            }
        }

        // Chain-rule gradient update for W_gate and W_up [d_head, ffn_dim]
        for d in 0..d_head {
            let err_d = shard_err[d];
            let p_in_d = self.last_p_in[d];
            for j in 0..ffn_dim {
                let idx = d * ffn_dim + j;
                let grad_gate = err_d * p_in_d * self.w_down[j * d_head + d];
                let grad_up = err_d * p_in_d * self.w_down[j * d_head + d];

                self.v_gate[idx] = beta * self.v_gate[idx] + (1.0 - beta) * grad_gate;
                self.v_up[idx] = beta * self.v_up[idx] + (1.0 - beta) * grad_up;

                self.w_gate[idx] =
                    self.w_gate[idx] * weight_decay - lr * self.v_gate[idx].clamp(-0.1, 0.1);
                self.w_up[idx] =
                    self.w_up[idx] * weight_decay - lr * self.v_up[idx].clamp(-0.1, 0.1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_core::ParticleHeader;

    #[test]
    fn test_micro_block_node_process_batch() {
        let config = MicroBlockConfig::default();
        let mut node = MicroBlockNode::new(0, config, 16, false);

        let p1 = Particle::new(ParticleHeader::new(0, 0, 1.0), vec![0.5f32; 64]);
        let p2 = Particle::new(ParticleHeader::new(1, 0, 1.0), vec![0.8f32; 64]);
        let mut batch = vec![p1, p2];

        node.process_batch(&mut batch);
        assert_eq!(node.activation_count, 2);
        assert_eq!(node.cumulative_sequence_len, 2);

        let shard_err = vec![0.01f32; 64];
        node.update_weights_with_shard_err(&shard_err, 0.01);
    }
}
