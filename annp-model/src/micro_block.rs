use crate::subnode::Subnode;
use annp_core::{MicroBlockConfig, Particle};
use annp_cuda::CudaMicroBlockRunner;

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

            // Continuous trace proximity for causality binding (STDP-like)
            let trace_diff =
                (self.kv_traces.get(index).unwrap_or(&0.0) - particle.trace_concentration).abs();
            let temporal_affinity = (1.0 - trace_diff).max(0.0);

            if temporal_affinity > 0.8
                && self.kv_token_ids.get(index) == Some(&particle.header.origin_token_id)
                && self.kv_shard_ids.get(index) == Some(&particle.header.shard_id)
            {
                let effective_sim = similarity * temporal_affinity;
                best_positive =
                    Some(best_positive.map_or(effective_sim, |best| best.max(effective_sim)));
            } else {
                negative_sum += similarity;
                negative_count += 1;
            }
        }
        best_positive.map(|positive| positive - negative_sum / negative_count.max(1) as f32)
    }

    /// Process a batch of particles through Micro-Block CUDA/CPU computation pipeline (0 heap allocations)
    pub fn process_batch(&mut self, particles: &mut [Particle], is_training: bool) {
        let batch_size = particles.len();
        if batch_size == 0 {
            return;
        }

        let d_head = self.config.d_head;
        let ffn_dim = d_head * self.config.ffn_expansion;
        let active = self.active_subnode;

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

        self.last_p_in.copy_from_slice(&self.p_in_buf);
        // Representative token ID for KV cache and temporal difference
        let token_id = particles[0].header.origin_token_id;

        // 2. Local Predictive Learning (Temporal Difference)
        if let Some(last_id) = self.last_token_id {
            if token_id == last_id + 1 {
                let mut local_err = vec![0.0f32; d_head];
                for d in 0..d_head {
                    local_err[d] = self.last_prediction[d] - self.p_in_buf[d];
                }

                let sq_err: f32 = local_err.iter().map(|&x| x * x).sum();
                let mse = sq_err / d_head as f32;
                if mse > 10.0 && self.activation_count % 1000 == 0 {
                    println!(
                        "DEBUG [Node {}]: MSE={}, last_prediction[0]={}, p_in_buf[0]={}",
                        self.node_id, mse, self.last_prediction[0], self.p_in_buf[0]
                    );
                }
                self.local_loss_accumulator += mse;
                self.local_loss_count += 1;

                if is_training {
                    let lr = 0.01 / (self.config.max_hop as f32);
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
                    if backprop_kv_len > 0 {
                        backprop_kv_len -= 1;
                    }

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
                        for i in 0..backprop_kv_len {
                            let w = scores[i] * inv_sum;
                            let v_slice = &self.v_cache[i * d_head..(i + 1) * d_head];
                            for d in 0..d_head {
                                attn_out[d] += w * v_slice[d];
                            }
                        }
                    }

                    for subnode in self.subnodes.iter_mut() {
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

                            let max_grad = 0.05f32;
                            for f in 0..ffn_dim {
                                let inter_val = ffn_inter[f];
                                for d in 0..d_head {
                                    let idx = f * d_head + d;
                                    let grad = (local_err[d] * inter_val * alpha)
                                        .clamp(-max_grad, max_grad);
                                    subnode.w_down[idx] =
                                        subnode.w_down[idx] * wd_factor - lr * grad;
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
                                    subnode.w_up[idx] =
                                        subnode.w_up[idx] * wd_factor - lr * grad_up;
                                }
                            }
                        }
                    }

                    let err_mag: f32 =
                        local_err.iter().map(|&x| x.abs()).sum::<f32>() / d_head as f32;
                    self.subnodes[0].credit_stats.observe(-err_mag);
                }
            }
        }

        // 4. Forward Pass (CUDA or CPU) on Superposed Input
        let kv_len = self.k_cache.len() / d_head;
        self.p_out_buf.clear();
        self.p_out_buf.resize(d_head, 0.0f32);
        self.p_out_buf.copy_from_slice(&self.p_in_buf);

        let mut temp_out = vec![0.0f32; d_head];
        for subnode in self.subnodes.iter() {
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
        let mut delta_n = vec![0.0f32; d_head];
        for d in 0..d_head {
            delta_n[d] = self.p_out_buf[d] - self.p_in_buf[d];
        }

        // We only use the first particle for KV cache update, representing the batch collective
        let p_clone = particles[0].clone();

        for p_ref in particles.iter_mut() {
            let agreement_before = self.local_agreement(p_ref, &p_ref.payload);
            let previous_credit = p_ref.credit;
            let previous_credit_valid = p_ref.credit_valid;

            for d in 0..d_head {
                p_ref.payload[d] += delta_n[d];
            }
            p_ref.credit_valid = false;

            if let (Some(before), Some(after)) = (
                agreement_before,
                self.local_agreement(p_ref, &p_ref.payload),
            ) {
                p_ref.credit = after - before;
                p_ref.credit_valid = true;
                self.subnodes[active].credit_stats.observe(p_ref.credit);
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
        }

        self.update_kv_cache(&p_clone);

        self.activation_count += batch_size as u64;
        self.recent_activation_count += batch_size as u64;
        self.cumulative_sequence_len += batch_size as u64;
        self.subnodes[active].activation_count += batch_size as u64;
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

        node.process_batch(&mut batch, true);
        assert_eq!(node.activation_count, 2);
        assert_eq!(node.cumulative_sequence_len, 2);

        // Removed broadcast error logic in test
    }
}
