use crate::subnode::Subnode;
use annp_core::{MicroBlockConfig, Particle};
use annp_cuda;

/// Autonomous Micro-Block Node (Container holding 1 to subnode_max Subnodes).
#[repr(align(128))]
pub struct MicroBlockNode {
    pub node_id: usize,
    pub config: MicroBlockConfig,
    pub subnodes: Vec<Subnode>,
    pub split_count: u32,
    // Local FIFO KV Cache
    pub k_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub v_cache: Vec<f32>, // Flat [kv_len * d_head]
    pub kv_traces: Vec<f32>,
    pub kv_token_ids: Vec<u32>,
    pub kv_shard_ids: Vec<u16>,
    pub max_kv_len: usize,
    // Last activation cache for exact chain-rule backpropagation
    pub last_p_in: Vec<f32>,       // [d_head]
    pub last_prediction: Vec<f32>, // [d_head]
    pub last_token_id: Option<u32>,
    // Node state statistics
    pub cumulative_sequence_len: u64, // S_j for plastic hardening
    pub activation_count: u64,
    pub local_loss_accumulator: f32,
    pub local_loss_count: usize,
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
    // Research Metrics
    pub sum_hop_count: u64,
    pub halted_particles_count: u64,
    pub sum_squared_energy: f32,
    pub total_particles_processed: u64,
    pub sum_attention_entropy: f32,
    pub attention_ops_count: u64,
    pub sum_credit_volatility: f32,
    pub sum_temporal_affinity: f32,
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

        let alpha_init = 1.0 / (d_head as f32).sqrt();
        let mut primary_subnode = Subnode::new_random(0, d_head, ffn_dim, alpha_init);
        if use_cuda {
            primary_subnode.d_weights = Some(annp_cuda::ffi::CudaDeviceWeights::new(
                &primary_subnode.w_gate,
                &primary_subnode.w_up,
                &primary_subnode.w_down,
            ));
        }
        let subnodes = vec![primary_subnode];

        let k_cache = Vec::with_capacity(max_kv_len * d_head);
        let v_cache = Vec::with_capacity(max_kv_len * d_head);
        let last_p_in = vec![0.0f32; d_head];

        let mut node = Self {
            node_id,
            config,
            subnodes,
            split_count: 0,
            k_cache,
            v_cache,
            kv_traces: Vec::with_capacity(max_kv_len),
            kv_token_ids: Vec::with_capacity(max_kv_len),
            kv_shard_ids: Vec::with_capacity(max_kv_len),
            max_kv_len,
            last_p_in,
            last_prediction: vec![0.0f32; d_head],
            last_token_id: None,
            local_loss_accumulator: 0.0,
            local_loss_count: 0,
            cumulative_sequence_len: 0,
            activation_count: 0,
            recent_activation_count: 0,
            p_in_buf: Vec::with_capacity(64 * d_head),
            p_out_buf: Vec::with_capacity(64 * d_head),
            use_cuda,
            active_subnode: 0,
            sum_hop_count: 0,
            halted_particles_count: 0,
            sum_squared_energy: 0.0,
            total_particles_processed: 0,
            sum_attention_entropy: 0.0,
            attention_ops_count: 0,
            sum_credit_volatility: 0.0,
            sum_temporal_affinity: 0.0,
        };
        node.sync_cuda_weights();
        node
    }

    pub fn sync_cuda_weights(&mut self) {
        if self.use_cuda {
            for subnode in self.subnodes.iter_mut() {
                if subnode.d_weights.is_none() {
                    subnode.d_weights = Some(annp_cuda::ffi::CudaDeviceWeights::new(
                        &subnode.w_gate,
                        &subnode.w_up,
                        &subnode.w_down,
                    ));
                }
            }
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
                self.kv_traces.remove(0);
                self.kv_token_ids.remove(0);
                self.kv_shard_ids.remove(0);
            }
        }

        let sq_sum: f32 = particle.payload.iter().map(|&x| x * x).sum();
        let inv_rms = 1.0 / (sq_sum / (d_head as f32) + 1e-8).sqrt();
        for &val in particle.payload.iter() {
            self.k_cache.push(val * inv_rms);
            self.v_cache.push(val * inv_rms);
        }
        self.kv_traces.push(particle.trace_concentration);
        self.kv_token_ids.push(particle.header.origin_token_id);
        self.kv_shard_ids.push(particle.header.shard_id);
        self.last_token_id = Some(particle.header.origin_token_id);
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
            let mut new_subnode = Subnode::spawn_from_parent(
                new_subnode_id,
                &parent_subnode,
                self.config.d_head,
                self.config.d_head * self.config.ffn_expansion,
            );
            if self.use_cuda {
                new_subnode.d_weights = Some(annp_cuda::ffi::CudaDeviceWeights::new(
                    &new_subnode.w_gate,
                    &new_subnode.w_up,
                    &new_subnode.w_down,
                ));
            }

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

    fn local_agreement(&self, particle: &Particle, payload: &[f32]) -> (Option<f32>, f32) {
        let d_head = self.config.d_head;
        let payload_norm = payload.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut best_positive = None::<f32>;
        let mut negative_sum = 0.0;
        let mut negative_count = 0u32;
        let mut temporal_affinity_sum = 0.0;
        let mut temporal_affinity_count = 0u32;
        for (index, key) in self.k_cache.chunks_exact(d_head).enumerate() {
            let key_norm = key.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            let similarity = payload.iter().zip(key).map(|(a, b)| a * b).sum::<f32>()
                / (payload_norm * key_norm);

            // Continuous trace proximity for causality binding (STDP-like)
            let trace_diff =
                (self.kv_traces.get(index).unwrap_or(&0.0) - particle.trace_concentration).abs();
            let temporal_affinity = (1.0 - trace_diff).max(0.0);

            // No hard-coded 0.8 gating; use smooth continuous temporal affinity scale
            if self.kv_token_ids.get(index) == Some(&particle.header.origin_token_id)
                && self.kv_shard_ids.get(index) == Some(&particle.header.shard_id)
            {
                let effective_sim = similarity * temporal_affinity;
                best_positive =
                    Some(best_positive.map_or(effective_sim, |best| best.max(effective_sim)));
                temporal_affinity_sum += temporal_affinity;
                temporal_affinity_count += 1;
            } else {
                negative_sum += similarity;
                negative_count += 1;
            }
        }
        let agreement =
            best_positive.map(|positive| positive - negative_sum / negative_count.max(1) as f32);
        let mean_ta = if temporal_affinity_count > 0 {
            temporal_affinity_sum / temporal_affinity_count as f32
        } else {
            0.0
        };
        (agreement, mean_ta)
    }

    /// Process a batch of particles through Micro-Block CUDA/CPU computation pipeline (0 heap allocations)
    pub fn process_batch(&mut self, particles: &mut [Particle], is_training: bool) {
        let batch_size = particles.len();
        if batch_size == 0 {
            return;
        }

        let trace_decay = 1.0 - 1.0 / (self.max_kv_len as f32).max(1.0);
        for trace in self.kv_traces.iter_mut() {
            *trace *= trace_decay;
        }

        let d_head = self.config.d_head;
        let ffn_dim = d_head * self.config.ffn_expansion;

        // --- UCB Multi-Armed Bandit Active Subnode Selection ---
        let total_activations: u64 = self.subnodes.iter().map(|s| s.credit_stats.count).sum();
        let ln_total = (total_activations as f32).ln().max(0.0);
        let mut best_ucb = f32::NEG_INFINITY;
        let mut active_idx = 0;

        for (i, subnode) in self.subnodes.iter().enumerate() {
            let ucb = if subnode.credit_stats.count == 0 {
                f32::INFINITY // Always explore newly born subnodes
            } else {
                let exploit = subnode.credit_stats.mean;
                let explore = (2.0 * ln_total / subnode.credit_stats.count as f32).sqrt();
                exploit + 0.1 * explore // C = 0.1
            };
            if ucb > best_ucb {
                best_ucb = ucb;
                active_idx = i;
            }
        }
        self.active_subnode = active_idx;
        let active = active_idx;

        // 1. Rigorous Superposition (Mean Field Aggregation)
        // By taking the mean, we preserve the variance of the field and make the node's
        // response permutation-invariant (order-independent) for simultaneous arrivals.
        self.p_in_buf.clear();
        self.p_in_buf.resize(d_head, 0.0);
        for p in particles.iter() {
            for d in 0..d_head {
                self.p_in_buf[d] += p.payload[d];
            }
        }
        let inv_n = 1.0 / (batch_size as f32);
        for d in 0..d_head {
            self.p_in_buf[d] *= inv_n;
        }

        // Representative token ID for KV cache and temporal difference
        let token_id = particles[0].header.origin_token_id;

        // 2. Local Predictive Learning (Temporal Difference)
        if let Some(last_id) = self.last_token_id
            && token_id == last_id + 1
        {
            let mut local_err = vec![0.0f32; d_head];
            for d in 0..d_head {
                local_err[d] = self.last_prediction[d] - self.p_in_buf[d];
            }

            let sq_err: f32 = local_err.iter().map(|&x| x * x).sum();
            let mse = sq_err / d_head as f32;
            self.local_loss_accumulator += mse;
            self.local_loss_count += 1;

            if is_training {
                // Mathematically scale learning rate directly by routing path expectation
                let lr = 1.0 / (self.config.max_hop as f32);
                let weight_decay = 1e-4f32;
                let wd_factor = 1.0 - lr * weight_decay;

                let kv_len = self.k_cache.len() / d_head;
                let mut p_in_normed = vec![0.0f32; d_head];
                let sq_sum_attn: f32 = self.last_p_in.iter().map(|&x| x * x).sum();
                let inv_rms_attn = 1.0 / (sq_sum_attn / (d_head as f32) + 1e-8).sqrt();
                for d in 0..d_head {
                    p_in_normed[d] = self.last_p_in[d] * inv_rms_attn;
                }

                let mut attn_out = vec![0.0f32; d_head];
                let mut backprop_kv_len = kv_len;
                backprop_kv_len = backprop_kv_len.saturating_sub(1);

                if backprop_kv_len > 0 {
                    let mut best_sim = -1e9f32;
                    let scale = 1.0 / (d_head as f32).sqrt();
                    let mut scores = vec![0.0f32; backprop_kv_len];

                    for i in 0..backprop_kv_len {
                        let k_slice = &self.k_cache[i * d_head..(i + 1) * d_head];
                        let mut dot = 0.0f32;
                        for d in 0..d_head {
                            dot += p_in_normed[d] * k_slice[d];
                        }
                        let score = dot * scale;
                        scores[i] = score;
                        if score > best_sim {
                            best_sim = score;
                        }
                    }

                    let mut sum_exp = 0.0f32;
                    for i in 0..backprop_kv_len {
                        let e = (scores[i] - best_sim).exp();
                        scores[i] = e;
                        sum_exp += e;
                    }

                    let inv_sum = 1.0 / (sum_exp + 1e-8);
                    let mut entropy = 0.0f32;
                    for i in 0..backprop_kv_len {
                        let w = scores[i] * inv_sum;
                        if w > 1e-10 {
                            entropy -= w * w.ln();
                        }
                        let v_slice = &self.v_cache[i * d_head..(i + 1) * d_head];
                        for d in 0..d_head {
                            attn_out[d] += w * v_slice[d];
                        }
                    }
                    self.sum_attention_entropy += entropy;
                    self.attention_ops_count += 1;
                }

                {
                    let subnode = &mut self.subnodes[active];
                    let alpha = subnode.alpha;

                    if self.use_cuda {
                        annp_cuda::ffi::CudaMicroBlockRunner::execute_backward(
                            &self.last_p_in,
                            &self.k_cache,
                            &self.v_cache,
                            &mut subnode.w_gate,
                            &mut subnode.w_up,
                            &mut subnode.w_down,
                            &local_err,
                            d_head,
                            ffn_dim,
                            backprop_kv_len,
                            alpha,
                            lr,
                            weight_decay,
                            None,
                            true,
                            subnode.d_weights.as_ref(),
                        );
                    } else {
                        // 3. CPU Fallback Backward Pass
                        let mut s_mid = vec![0.0f32; d_head];
                        for d in 0..d_head {
                            s_mid[d] = self.last_p_in[d] + alpha * attn_out[d];
                        }

                        let mut s_mid_normed = vec![0.0f32; d_head];
                        let sq_sum_ffn: f32 = s_mid.iter().map(|&x| x * x).sum();
                        let inv_rms_ffn = 1.0 / (sq_sum_ffn / (d_head as f32) + 1e-8).sqrt();
                        for d in 0..d_head {
                            s_mid_normed[d] = s_mid[d] * inv_rms_ffn;
                        }

                        let mut ffn_inter = vec![0.0f32; ffn_dim];
                        let mut gate_arr = vec![0.0f32; ffn_dim];
                        let mut up_arr = vec![0.0f32; ffn_dim];
                        let mut swish_arr = vec![0.0f32; ffn_dim];

                        for f in 0..ffn_dim {
                            let mut gate = 0.0f32;
                            let mut up = 0.0f32;
                            for d in 0..d_head {
                                let m_val = s_mid_normed[d];
                                gate += m_val * subnode.w_gate[d * ffn_dim + f];
                                up += m_val * subnode.w_up[d * ffn_dim + f];
                            }

                            let sig = 1.0 / (1.0 + (-gate).exp());
                            let swish = gate * sig;

                            gate_arr[f] = gate;
                            up_arr[f] = up;
                            swish_arr[f] = swish;
                            ffn_inter[f] = swish * up;
                        }

                        let mut d_inter = vec![0.0f32; ffn_dim];
                        for f in 0..ffn_dim {
                            let mut sum = 0.0f32;
                            for d in 0..d_head {
                                sum += local_err[d] * subnode.w_down[f * d_head + d];
                            }
                            d_inter[f] = sum;
                        }

                        let mut d_gate_arr = vec![0.0f32; ffn_dim];
                        let mut d_up_arr = vec![0.0f32; ffn_dim];
                        for f in 0..ffn_dim {
                            let d_int = d_inter[f];
                            let d_swish = d_int * up_arr[f];
                            let d_up = d_int * swish_arr[f];

                            let gate = gate_arr[f];
                            let sig = 1.0 / (1.0 + (-gate).exp());
                            let d_sig_d_gate = sig * (1.0 - sig);
                            let d_swish_d_gate = sig + gate * d_sig_d_gate;

                            let d_gate = d_swish * d_swish_d_gate;
                            d_gate_arr[f] = d_gate;
                            d_up_arr[f] = d_up;
                        }

                        // Adaptive local gradient clipping scale rather than arbitrary 0.05
                        let max_grad = 1.0 / (d_head as f32).sqrt();
                        for f in 0..ffn_dim {
                            let inter_val = ffn_inter[f];
                            for d in 0..d_head {
                                let idx = f * d_head + d;
                                let grad =
                                    (local_err[d] * inter_val * alpha).clamp(-max_grad, max_grad);
                                subnode.w_down[idx] = subnode.w_down[idx] * wd_factor - lr * grad;
                            }
                        }

                        for d in 0..d_head {
                            let m_val = s_mid_normed[d];
                            for f in 0..ffn_dim {
                                let idx = d * ffn_dim + f;
                                let grad_gate =
                                    (d_gate_arr[f] * m_val * alpha).clamp(-max_grad, max_grad);
                                let grad_up =
                                    (d_up_arr[f] * m_val * alpha).clamp(-max_grad, max_grad);

                                subnode.w_gate[idx] =
                                    subnode.w_gate[idx] * wd_factor - lr * grad_gate;
                                subnode.w_up[idx] = subnode.w_up[idx] * wd_factor - lr * grad_up;
                            }
                        }
                    }
                }

                let err_mag: f32 = local_err.iter().map(|&x| x.abs()).sum::<f32>() / d_head as f32;
                self.subnodes[active].credit_stats.observe(-err_mag);
            }
        }

        self.last_p_in.copy_from_slice(&self.p_in_buf);

        // 4. Forward Pass (CUDA or CPU) on Superposed Input
        let kv_len = self.k_cache.len() / d_head;
        self.p_out_buf.clear();
        self.p_out_buf.resize(d_head, 0.0f32);
        self.p_out_buf.copy_from_slice(&self.p_in_buf);

        let mut temp_out = vec![0.0f32; d_head];
        {
            let subnode = &self.subnodes[active];
            let alpha = subnode.alpha;
            let w_gate = &subnode.w_gate;
            let w_up = &subnode.w_up;
            let w_down = &subnode.w_down;

            annp_cuda::ffi::CudaMicroBlockRunner::execute_fused_with_stream_device(
                &self.p_in_buf,
                &self.k_cache,
                &self.v_cache,
                w_gate,
                w_up,
                w_down,
                &mut temp_out,
                1,
                d_head,
                ffn_dim,
                kv_len,
                alpha,
                None,
                self.use_cuda,
                subnode.d_weights.as_ref(),
            );

            for d in 0..d_head {
                self.p_out_buf[d] += temp_out[d] - self.p_in_buf[d];
            }
        }

        self.last_prediction.copy_from_slice(&self.p_out_buf);
        self.last_token_id = Some(token_id);

        // 5. Apply Transformation to All Particles
        let active_alpha = self.subnodes[active].alpha;
        let scale_factor = 1.0 / (1.0 + active_alpha * active_alpha).sqrt();
        let mut delta_n = vec![0.0f32; d_head];
        for d in 0..d_head {
            delta_n[d] = self.p_out_buf[d] - self.p_in_buf[d];
        }

        // We only use the first particle for KV cache update, representing the batch collective
        let p_clone = particles[0].clone();

        for p_ref in particles.iter_mut() {
            let (agreement_before, ta_before) = self.local_agreement(p_ref, &p_ref.payload);
            let previous_credit = p_ref.credit;
            let previous_credit_valid = p_ref.credit_valid;

            for d in 0..d_head {
                let new_val = (p_ref.payload[d] + delta_n[d]) * scale_factor;
                p_ref.payload[d] = new_val;
            }
            p_ref.credit_valid = false;

            let (agreement_after, ta_after) = self.local_agreement(p_ref, &p_ref.payload);
            self.sum_temporal_affinity += (ta_before + ta_after) * 0.5;

            if let (Some(before), Some(after)) = (agreement_before, agreement_after) {
                p_ref.credit = after - before;
                p_ref.credit_valid = true;
                self.subnodes[active].credit_stats.observe(p_ref.credit);
                self.sum_credit_volatility += p_ref.credit.abs();
            }

            p_ref.header.step_hop(self.config.max_hop);

            if !p_ref.header.halted
                && previous_credit_valid
                && p_ref.credit_valid
                && previous_credit <= 0.0
                && p_ref.credit <= 0.0
            {
                p_ref.header.halted = true;
            }

            self.total_particles_processed += 1;
            if p_ref.header.halted {
                self.halted_particles_count += 1;
            }
            self.sum_hop_count += p_ref.header.hop_count as u64;

            let energy: f32 = p_ref.payload.iter().map(|&x| x * x).sum();
            self.sum_squared_energy += energy;
        }

        self.update_kv_cache(&p_clone);

        self.activation_count += batch_size as u64;
        self.recent_activation_count += batch_size as u64;
        self.cumulative_sequence_len += batch_size as u64;
        self.subnodes[active].activation_count += batch_size as u64;

        self.try_subnode_neurogenesis();
        self.prune_dominated_subnodes();
    }

    /// Extract current research metrics and reset node-local accumulators
    pub fn extract_and_reset_metrics(&mut self) -> NodeMetrics {
        let metrics = NodeMetrics {
            sum_hop_count: self.sum_hop_count,
            halted_particles_count: self.halted_particles_count,
            sum_squared_energy: self.sum_squared_energy,
            total_particles_processed: self.total_particles_processed,
            sum_attention_entropy: self.sum_attention_entropy,
            attention_ops_count: self.attention_ops_count,
            sum_credit_volatility: self.sum_credit_volatility,
            sum_temporal_affinity: self.sum_temporal_affinity,
            active_subnodes_count: self.subnodes.len() as u64,
        };

        self.sum_hop_count = 0;
        self.halted_particles_count = 0;
        self.sum_squared_energy = 0.0;
        self.total_particles_processed = 0;
        self.sum_attention_entropy = 0.0;
        self.attention_ops_count = 0;
        self.sum_credit_volatility = 0.0;
        self.sum_temporal_affinity = 0.0;

        metrics
    }
}

pub struct NodeMetrics {
    pub sum_hop_count: u64,
    pub halted_particles_count: u64,
    pub sum_squared_energy: f32,
    pub total_particles_processed: u64,
    pub sum_attention_entropy: f32,
    pub attention_ops_count: u64,
    pub sum_credit_volatility: f32,
    pub sum_temporal_affinity: f32,
    pub active_subnodes_count: u64,
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

        node.process_batch(&mut batch, true);
        assert_eq!(node.activation_count, 2);
        assert_eq!(node.cumulative_sequence_len, 2);

        // Removed broadcast error logic in test
    }
}
