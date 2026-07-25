use annp_core::{compute_attention_entropy, compute_delta_p, MicroBlockConfig, NormStrategy, Particle};
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
    // MicroNorm learnable parameter
    pub alpha: f32,
    // Local FIFO KV Cache
    pub k_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub v_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub max_kv_len: usize,
    // Node state statistics
    pub cumulative_sequence_len: u64, // S_j for plastic hardening
    pub activation_count: u64,
}

impl MicroBlockNode {
    pub fn new(node_id: usize, config: MicroBlockConfig, max_kv_len: usize) -> Self {
        let d_head = config.d_head;
        let ffn_dim = d_head * config.ffn_expansion;

        let mut rng = rand::thread_rng();
        let scale = (2.0 / (d_head + ffn_dim) as f64).sqrt() as f32;

        let w_gate = (0..d_head * ffn_dim).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_up = (0..d_head * ffn_dim).map(|_| rng.gen_range(-scale..scale)).collect();
        let w_down = (0..ffn_dim * d_head).map(|_| rng.gen_range(-scale..scale)).collect();

        Self {
            node_id,
            alpha: config.alpha_init,
            config,
            w_gate,
            w_up,
            w_down,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            max_kv_len,
            cumulative_sequence_len: 0,
            activation_count: 0,
        }
    }

    /// Update KV Cache with new incoming particle (push FIFO)
    pub fn update_kv_cache(&mut self, particle: &Particle) {
        let d_head = self.config.d_head;
        assert_eq!(particle.payload.len(), d_head);

        let curr_len = self.k_cache.len() / d_head;
        if curr_len >= self.max_kv_len {
            // Evict oldest KV entry
            self.k_cache.drain(0..d_head);
            self.v_cache.drain(0..d_head);
        }

        self.k_cache.extend_from_slice(&particle.payload);
        self.v_cache.extend_from_slice(&particle.payload);
    }

    /// Process a batch of particles through Micro-Block CUDA/CPU computation pipeline
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

        // Flatten p_in payloads
        let mut p_in_flat = Vec::with_capacity(batch_size * d_head);
        for p in particles.iter() {
            p_in_flat.extend_from_slice(&p.payload);
        }

        let mut p_out_flat = vec![0.0f32; batch_size * d_head];

        // Launch CUDA / Fused CudaMicroBlockRunner kernel
        CudaMicroBlockRunner::execute_fused(
            &p_in_flat,
            &self.k_cache,
            &self.v_cache,
            &self.w_gate,
            &self.w_up,
            &self.w_down,
            &mut p_out_flat,
            batch_size,
            d_head,
            ffn_dim,
            kv_len,
            norm_strat_val,
            self.alpha,
            self.config.sphere_radius,
        );

        // Update particles, evaluation halting condition and metrics
        for (i, p) in particles.iter_mut().enumerate() {
            let out_slice = &p_out_flat[i * d_head..(i + 1) * d_head];
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
}
