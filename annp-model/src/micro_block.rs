use crate::subnode::Subnode;
use annp_core::{MicroBlockConfig, NormStrategy, Particle};
use annp_cuda::CudaMicroBlockRunner;

/// Autonomous Micro-Block Node (Container holding 1 to subnode_max Subnodes).
pub struct MicroBlockNode {
    pub node_id: usize,
    pub config: MicroBlockConfig,
    pub subnodes: Vec<Subnode>,
    pub split_count: u32,
    // Local FIFO KV Cache
    pub k_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub v_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub kv_wave_ids: Vec<u64>,
    pub kv_token_ids: Vec<u32>,
    pub kv_shard_ids: Vec<u16>,
    pub max_kv_len: usize,
    // Last activation cache for exact chain-rule backpropagation
    pub last_p_in: Vec<f32>, // [d_head]
    // Node state statistics
    pub cumulative_sequence_len: u64, // S_j for plastic hardening
    pub activation_count: u64,
    /// Reset at each wave. Only nodes that actually participated in the most
    /// recent local dynamics may receive that wave's training signal.
    pub recent_activation_count: u64,
    // Reusable workspace scratch buffers to avoid heap allocations in process_batch
    pub p_in_buf: Vec<f32>,
    pub p_out_buf: Vec<f32>,
    pub use_cuda: bool,
    /// The currently selected local expert.  Neurogenesis is meaningful only
    /// when newly spawned subnodes can participate in subsequent dynamics.
    pub active_subnode: usize,
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

        let primary_subnode = Subnode::new_random(0, d_head, ffn_dim, config.alpha_init);
        let subnodes = vec![primary_subnode];

        let k_cache = Vec::with_capacity(max_kv_len * d_head);
        let v_cache = Vec::with_capacity(max_kv_len * d_head);
        let last_p_in = vec![0.0f32; d_head];

        Self {
            node_id,
            config,
            subnodes,
            split_count: 0,
            k_cache,
            v_cache,
            kv_wave_ids: Vec::with_capacity(max_kv_len),
            kv_token_ids: Vec::with_capacity(max_kv_len),
            kv_shard_ids: Vec::with_capacity(max_kv_len),
            max_kv_len,
            last_p_in,
            cumulative_sequence_len: 0,
            activation_count: 0,
            recent_activation_count: 0,
            p_in_buf: Vec::with_capacity(64 * d_head),
            p_out_buf: Vec::with_capacity(64 * d_head),
            use_cuda,
            active_subnode: 0,
        }
    }

    /// Primary active subnode reference
    pub fn primary_subnode(&self) -> &Subnode {
        &self.subnodes[0]
    }

    /// Primary active subnode mutable reference
    pub fn primary_subnode_mut(&mut self) -> &mut Subnode {
        &mut self.subnodes[0]
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
                self.kv_wave_ids.remove(0);
                self.kv_token_ids.remove(0);
                self.kv_shard_ids.remove(0);
            }
        }

        self.k_cache.extend_from_slice(&particle.payload);
        self.v_cache.extend_from_slice(&particle.payload);
        self.kv_wave_ids.push(particle.wave_id);
        self.kv_token_ids.push(particle.header.origin_token_id);
        self.kv_shard_ids.push(particle.header.shard_id);
    }

    /// Spawn only when the active local dynamics have empirical evidence of no
    /// improvement. This removes the activation-count threshold from growth.
    pub fn try_subnode_neurogenesis(&mut self) -> bool {
        let active = &self.subnodes[self.active_subnode].credit_stats;
        if self.subnodes.len() < self.config.subnode_max
            && active.count > 1
            && active.optimistic_value() <= 0.0
        {
            let parent_subnode = self
                .subnodes
                .last()
                .cloned()
                .unwrap_or_else(|| self.subnodes[0].clone());
            let new_subnode_id = self.subnodes.len();
            let new_subnode = Subnode::spawn_from_parent(new_subnode_id, &parent_subnode, 1e-3);

            self.subnodes.push(new_subnode);
            self.split_count += 1;
            true
        } else {
            false
        }
    }

    /// Retire an expert only when its optimistic local credit is below another
    /// expert's pessimistic credit. Candidate lifecycle therefore needs no
    /// trial-length or quality threshold.
    pub fn prune_dominated_subnodes(&mut self) -> usize {
        if self.subnodes.len() <= 1 {
            return 0;
        }
        let best_lower = self
            .subnodes
            .iter()
            .map(|subnode| subnode.credit_stats.pessimistic_value())
            .fold(f32::NEG_INFINITY, f32::max);
        let keep: Vec<usize> = self
            .subnodes
            .iter()
            .enumerate()
            .filter_map(|(index, subnode)| {
                (subnode.credit_stats.optimistic_value() >= best_lower).then_some(index)
            })
            .collect();
        if keep.len() == self.subnodes.len() || keep.is_empty() {
            return 0;
        }
        let old_len = self.subnodes.len();
        let active_old = self.active_subnode;
        self.subnodes = keep
            .iter()
            .map(|&index| self.subnodes[index].clone())
            .collect();
        self.active_subnode = keep
            .iter()
            .position(|&index| index == active_old)
            .unwrap_or(0);
        self.subnodes
            .iter_mut()
            .enumerate()
            .for_each(|(index, subnode)| subnode.subnode_id = index);
        old_len - self.subnodes.len()
    }

    fn select_active_subnode(&mut self) {
        if self.subnodes.len() <= 1 {
            self.active_subnode = 0;
            return;
        }
        // A new expert is automatically explored because its uncertainty is
        // infinite; established experts compete by local credit only.
        self.active_subnode = self
            .subnodes
            .iter()
            .enumerate()
            .map(|(index, subnode)| (index, subnode.credit_stats.optimistic_value()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
    }

    fn local_agreement(&self, particle: &Particle, payload: &[f32]) -> Option<f32> {
        let d_head = self.config.d_head;
        let payload_norm = payload.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut best_positive = None::<f32>;
        let mut negative_sum = 0.0;
        let mut negative_count = 0u32;
        for (index, key) in self.k_cache.chunks_exact(d_head).enumerate() {
            let key_norm = key.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            let similarity = payload.iter().zip(key).map(|(a, b)| a * b).sum::<f32>()
                / (payload_norm * key_norm);
            if self.kv_wave_ids.get(index) == Some(&particle.wave_id)
                && self.kv_token_ids.get(index) == Some(&particle.header.origin_token_id)
                && self.kv_shard_ids.get(index) == Some(&particle.header.shard_id)
            {
                best_positive = Some(best_positive.map_or(similarity, |best| best.max(similarity)));
            } else {
                negative_sum += similarity;
                negative_count += 1;
            }
        }
        best_positive.map(|positive| positive - negative_sum / negative_count.max(1) as f32)
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
        self.select_active_subnode();

        for p in particles.iter() {
            self.p_in_buf.extend_from_slice(&p.payload);
        }

        self.p_out_buf.clear();
        self.p_out_buf.resize(batch_size * d_head, 0.0f32);

        // Compute through primary active subnode
        let active = self.active_subnode;
        let alpha = self.subnodes[active].alpha;
        let w_gate = &self.subnodes[active].w_gate;
        let w_up = &self.subnodes[active].w_up;
        let w_down = &self.subnodes[active].w_down;

        // Launch CUDA / Fused CudaMicroBlockRunner kernel (respecting self.use_cuda flag)
        CudaMicroBlockRunner::execute_fused_with_stream_device(
            &self.p_in_buf,
            &self.k_cache,
            &self.v_cache,
            w_gate,
            w_up,
            w_down,
            &mut self.p_out_buf,
            batch_size,
            d_head,
            ffn_dim,
            kv_len,
            norm_strat_val,
            alpha,
            self.config.sphere_radius,
            None,
            self.use_cuda,
        );

        // Update particles, evaluation halting condition and metrics
        for (i, p) in particles.iter_mut().enumerate() {
            let out_slice = &self.p_out_buf[i * d_head..(i + 1) * d_head];
            let agreement_before = self.local_agreement(p, &p.payload);
            let previous_credit = p.credit;
            let previous_credit_valid = p.credit_valid;

            p.payload.copy_from_slice(out_slice);
            p.credit_valid = false;
            if let (Some(before), Some(after)) =
                (agreement_before, self.local_agreement(p, &p.payload))
            {
                p.credit = after - before;
                p.credit_valid = true;
                self.subnodes[active].credit_stats.observe(p.credit);
            }
            p.header.step_hop(self.config.max_hop);

            // Two successive non-improving local transitions are sufficient to
            // settle; this is scale-free and needs no epsilon or entropy cutoff.
            if !p.header.halted
                && previous_credit_valid
                && p.credit_valid
                && previous_credit <= 0.0
                && p.credit <= 0.0
            {
                p.header.halted = true;
            }

            // Update KV Cache with computed output
            self.update_kv_cache(p);
        }

        self.activation_count += batch_size as u64;
        self.recent_activation_count += batch_size as u64;
        self.cumulative_sequence_len += batch_size as u64;
        self.subnodes[active].activation_count += batch_size as u64;
    }

    /// Chain-Rule Backpropagation Update Step (v = 0.9 * v + 0.1 * g)
    pub fn update_weights_with_shard_err(&mut self, shard_err: &[f32], lr: f32) {
        let d_head = self.config.d_head;
        if shard_err.len() < d_head || self.recent_activation_count == 0 {
            return;
        }

        let ffn_dim = d_head * self.config.ffn_expansion;
        let primary = &mut self.subnodes[self.active_subnode];

        let ffn_scale = (ffn_dim as f32).sqrt();
        let head_scale = (d_head as f32).sqrt();

        // Chain-rule gradient update for W_down [ffn_dim, d_head]
        for j in 0..ffn_dim {
            let p_in_val = self.last_p_in[j % d_head];
            for d in 0..d_head {
                let idx = j * d_head + d;
                let grad = (shard_err[d] * p_in_val) / ffn_scale;
                primary.w_down[idx] -= lr * grad;
            }
        }

        // Chain-rule gradient update for W_gate and W_up [d_head, ffn_dim]
        for d in 0..d_head {
            let err_d = shard_err[d];
            let p_in_d = self.last_p_in[d];
            for j in 0..ffn_dim {
                let idx = d * ffn_dim + j;
                let grad_gate = (err_d * p_in_d * primary.w_down[j * d_head + d]) / head_scale;
                let grad_up = (err_d * p_in_d * primary.w_down[j * d_head + d]) / head_scale;

                primary.w_gate[idx] -= lr * grad_gate;
                primary.w_up[idx] -= lr * grad_up;
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
