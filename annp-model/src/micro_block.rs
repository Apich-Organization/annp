use crate::subnode::Subnode;
use annp_core::{MicroBlockConfig, Particle};
use annp_cuda;

#[repr(align(128))]
pub struct MicroBlockNode {
    pub node_id: usize,
    pub config: MicroBlockConfig,
    pub subnodes: Vec<Subnode>,
    pub split_count: u32,

    // Local Implicit Memory (Fast Weights)
    pub fast_weight: Vec<f32>, // Flat [d_head * d_head]
    pub cumulative_energy: f32,

    // Last activation cache for exact chain-rule backpropagation
    pub last_p_in: Vec<f32>,       // [d_head]
    pub last_prediction: Vec<f32>, // [d_head]
    pub last_token_id: Option<u32>,

    // Node state statistics
    pub cumulative_sequence_len: u64, // S_j for plastic hardening
    pub activation_count: u64,
    pub local_loss_accumulator: f32,
    pub local_loss_count: usize,
    pub recent_activation_count: u64,

    // Reusable workspace scratch buffers
    pub p_in_buf: Vec<f32>,
    pub p_out_buf: Vec<f32>,
    pub use_cuda: bool,
    pub active_subnode: usize,

    // Research Metrics
    pub sum_hop_count: u64,
    pub halted_particles_count: u64,
    pub sum_squared_energy: f32,
    pub total_particles_processed: u64,
    pub sum_memory_density: f32,
    pub attention_ops_count: u64,
    pub sum_credit_volatility: f32,
    pub sum_temporal_affinity: f32,
}

impl MicroBlockNode {
    pub fn new(node_id: usize, config: MicroBlockConfig, use_cuda: bool) -> Self {
        let d_head = config.d_head;
        let ffn_dim = d_head * config.ffn_expansion;

        let alpha_init = 1.0 / (d_head as f32).sqrt();
        let memory_size = (d_head * d_head) as f32;
        let gamma = 1.0 - 1.0 / memory_size.max(1.0);

        let mut primary_subnode = Subnode::new_random(0, d_head, ffn_dim, alpha_init, gamma);
        if use_cuda {
            primary_subnode.d_weights = Some(annp_cuda::ffi::CudaDeviceWeights::new(
                &primary_subnode.w_gate,
                &primary_subnode.w_up,
                &primary_subnode.w_down,
            ));
        }
        let subnodes = vec![primary_subnode];

        let fast_weight = vec![0.0f32; d_head * d_head];
        let last_p_in = vec![0.0f32; d_head];

        let mut node = Self {
            node_id,
            config,
            subnodes,
            split_count: 0,
            fast_weight,
            cumulative_energy: 0.0,
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
            sum_memory_density: 0.0,
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

    pub fn primary_subnode(&self) -> &Subnode {
        &self.subnodes[0]
    }

    pub fn primary_subnode_mut(&mut self) -> &mut Subnode {
        &mut self.subnodes[0]
    }

    pub fn try_subnode_neurogenesis(&mut self) -> bool {
        let active_idx = self.active_subnode;
        let active = &self.subnodes[active_idx];

        let loss_avg = if self.local_loss_count > 0 {
            self.local_loss_accumulator / self.local_loss_count as f32
        } else {
            f32::INFINITY
        };

        let health_threshold = 1.0 + self.subnodes.len() as f32;

        if self.subnodes.len() < self.config.subnode_max
            && active.health > health_threshold
            && active.credit_stats.variance() > loss_avg
        {
            let parent_subnode = active.clone();
            let new_subnode_id = self.subnodes.len();
            let mut new_subnode = Subnode::spawn_from_parent(
                new_subnode_id,
                &parent_subnode,
                self.config.d_head,
                self.config.d_head * self.config.ffn_expansion,
                1.0 - 1.0 / (self.config.d_head * self.config.d_head) as f32,
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

            self.subnodes[active_idx].health *= 0.5;
            true
        } else {
            false
        }
    }

    pub fn prune_dominated_subnodes(&mut self) -> usize {
        if self.subnodes.len() <= 1 {
            return 0;
        }

        let keep: Vec<usize> = self
            .subnodes
            .iter()
            .enumerate()
            .filter_map(|(index, subnode)| (subnode.health > 0.0).then_some(index))
            .collect();

        if keep.len() == self.subnodes.len() {
            return 0;
        }

        if keep.is_empty() {
            let best_idx = self
                .subnodes
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.health.partial_cmp(&b.1.health).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.subnodes[best_idx].health = 0.5;
            let keep = vec![best_idx];

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
            return old_len - self.subnodes.len();
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

    pub fn process_batch(&mut self, particles: &mut [Particle], learning_rate: Option<f32>) {
        let batch_size = particles.len();
        if batch_size == 0 {
            return;
        }

        let d_head = self.config.d_head;
        let ffn_dim = d_head * self.config.ffn_expansion;

        self.p_in_buf.clear();
        self.p_in_buf.resize(d_head, 0.0);
        let mut batch_energy = 0.0f32;
        for p in particles.iter() {
            let mut energy = 0.0;
            for d in 0..d_head {
                self.p_in_buf[d] += p.payload[d];
                energy += p.payload[d] * p.payload[d];
            }
            batch_energy += energy;
        }
        self.cumulative_energy += batch_energy;
        let inv_n = 1.0 / (batch_size as f32);
        for d in 0..d_head {
            self.p_in_buf[d] *= inv_n;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut active_idx = 0;

        let decay = 1.0 / (d_head as f32 * self.subnodes.len().max(1) as f32);
        for (i, subnode) in self.subnodes.iter_mut().enumerate() {
            subnode.health -= decay;

            let score = if subnode.credit_stats.count <= 1.0 {
                f32::INFINITY
            } else {
                let mean = subnode.credit_stats.mean;
                let se = subnode.credit_stats.standard_error();
                let df = (subnode.credit_stats.count - 1.0).max(1.0);

                let t_sample = annp_core::student_t_sample_approximation(df);
                mean + se * t_sample
            };
            if score > best_score {
                best_score = score;
                active_idx = i;
            }
        }
        self.active_subnode = active_idx;
        let active = active_idx;
        let recovery = self.subnodes[active].alpha / (d_head as f32);
        self.subnodes[active].health += recovery;

        let token_id = particles[0].header.origin_token_id;

        if let Some(last_id) = self.last_token_id {
            if token_id == last_id + 1 {
                let mut local_err = vec![0.0f32; d_head];
                for (err, (pred, p_in)) in local_err
                    .iter_mut()
                    .zip(self.last_prediction.iter().zip(self.p_in_buf.iter()))
                {
                    *err = pred - p_in;
                }

                let sq_err: f32 = local_err.iter().map(|&x| x * x).sum();
                let mse = sq_err / d_head as f32;
                self.local_loss_accumulator += mse;
                self.local_loss_count += 1;

                if let Some(lr) = learning_rate {
                    let weight_decay = self.config.weight_decay;
                    let wd_factor = 1.0 - lr * weight_decay;

                    let mut p_in_normed = vec![0.0f32; d_head];
                    let sq_sum_attn: f32 = self.last_p_in.iter().map(|&x| x * x).sum();
                    let inv_rms_attn = 1.0 / (sq_sum_attn / (d_head as f32) + 1e-8).sqrt();
                    for (normed, &in_val) in p_in_normed.iter_mut().zip(self.last_p_in.iter()) {
                        *normed = in_val * inv_rms_attn;
                    }

                    let lambda = 1.0 - 1.0 / self.cumulative_energy.max(1.0).sqrt();
                    let max_grad = 1.0 / (d_head as f32).sqrt();
                    for r in 0..d_head {
                        for c in 0..d_head {
                            let idx = r * d_head + c;
                            let grad = (local_err[r] * p_in_normed[c]).clamp(-max_grad, max_grad);
                            self.fast_weight[idx] = self.fast_weight[idx] * lambda - lr * grad;
                        }
                    }

                    let mut attn_out = vec![0.0f32; d_head];
                    for r in 0..d_head {
                        let mut sum = 0.0;
                        for c in 0..d_head {
                            sum += self.fast_weight[r * d_head + c] * p_in_normed[c];
                        }
                        attn_out[r] = sum;
                    }

                    {
                        let subnode = &mut self.subnodes[active];
                        let alpha = subnode.alpha;

                        let mut s_mid = vec![0.0f32; d_head];
                        for (s, (&in_val, &attn_val)) in s_mid
                            .iter_mut()
                            .zip(self.last_p_in.iter().zip(attn_out.iter()))
                        {
                            *s = in_val + alpha * attn_val;
                        }

                        let mut s_mid_normed = vec![0.0f32; d_head];
                        let sq_sum_ffn: f32 = s_mid.iter().map(|&x| x * x).sum();
                        let inv_rms_ffn = 1.0 / (sq_sum_ffn / (d_head as f32) + 1e-8).sqrt();
                        for (normed, &s_val) in s_mid_normed.iter_mut().zip(s_mid.iter()) {
                            *normed = s_val * inv_rms_ffn;
                        }

                        let mut ffn_inter = vec![0.0f32; ffn_dim];
                        let mut gate_arr = vec![0.0f32; ffn_dim];
                        let mut up_arr = vec![0.0f32; ffn_dim];
                        let mut swish_arr = vec![0.0f32; ffn_dim];

                        for f in 0..ffn_dim {
                            let mut gate = 0.0f32;
                            let mut up = 0.0f32;
                            for (d, &m_val) in s_mid_normed.iter().enumerate() {
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
                        for (f, inter_val) in d_inter.iter_mut().enumerate() {
                            let mut sum = 0.0f32;
                            for (d, &err) in local_err.iter().enumerate() {
                                sum += err * subnode.w_down[f * d_head + d];
                            }
                            *inter_val = sum;
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

                        let max_grad = 1.0 / (d_head as f32).sqrt();
                        for (f, &inter_val) in ffn_inter.iter().enumerate() {
                            let w_down_slice = &mut subnode.w_down[f * d_head..(f + 1) * d_head];
                            for (d, &err_val) in local_err.iter().enumerate() {
                                let grad = (err_val * inter_val * alpha).clamp(-max_grad, max_grad);
                                w_down_slice[d] = w_down_slice[d] * wd_factor - lr * grad;
                            }
                        }

                        for (d, &m_val) in s_mid_normed.iter().enumerate() {
                            let w_gate_slice = &mut subnode.w_gate[d * ffn_dim..(d + 1) * ffn_dim];
                            let w_up_slice = &mut subnode.w_up[d * ffn_dim..(d + 1) * ffn_dim];
                            for f in 0..ffn_dim {
                                let grad_gate =
                                    (d_gate_arr[f] * m_val * alpha).clamp(-max_grad, max_grad);
                                let grad_up =
                                    (d_up_arr[f] * m_val * alpha).clamp(-max_grad, max_grad);

                                w_gate_slice[f] = w_gate_slice[f] * wd_factor - lr * grad_gate;
                                w_up_slice[f] = w_up_slice[f] * wd_factor - lr * grad_up;
                            }
                        }
                    }
                }
            }
        }

        let mut dot = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        for (&in_val, &last_val) in self.p_in_buf.iter().zip(self.last_p_in.iter()) {
            dot += in_val * last_val;
            norm1 += in_val * in_val;
            norm2 += last_val * last_val;
        }
        let affinity = if norm1 > 0.0 && norm2 > 0.0 {
            dot / (norm1 * norm2).sqrt()
        } else {
            0.0
        };
        self.sum_temporal_affinity += affinity * batch_size as f32;

        self.last_p_in.copy_from_slice(&self.p_in_buf);

        self.p_out_buf.clear();
        self.p_out_buf.resize(d_head, 0.0f32);
        self.p_out_buf.copy_from_slice(&self.p_in_buf);

        let mut p_in_normed = vec![0.0f32; d_head];
        let sq_sum_in: f32 = self.p_in_buf.iter().map(|&x| x * x).sum();
        let inv_rms_in = 1.0 / (sq_sum_in / (d_head as f32) + 1e-8).sqrt();
        for (normed, &in_val) in p_in_normed.iter_mut().zip(self.p_in_buf.iter()) {
            *normed = in_val * inv_rms_in;
        }

        let mut attn_out = vec![0.0f32; d_head];
        for r in 0..d_head {
            let mut sum = 0.0;
            for c in 0..d_head {
                sum += self.fast_weight[r * d_head + c] * p_in_normed[c];
            }
            attn_out[r] = sum;
        }

        let mut temp_out = vec![0.0f32; d_head];
        {
            let subnode = &self.subnodes[active];
            let alpha = subnode.alpha;

            let mut s_mid = vec![0.0f32; d_head];
            for (s, (&in_val, &a_val)) in s_mid
                .iter_mut()
                .zip(self.p_in_buf.iter().zip(attn_out.iter()))
            {
                *s = in_val + alpha * a_val;
            }

            let mut s_mid_normed = vec![0.0f32; d_head];
            let sq_sum_ffn: f32 = s_mid.iter().map(|&x| x * x).sum();
            let inv_rms_ffn = 1.0 / (sq_sum_ffn / (d_head as f32) + 1e-8).sqrt();
            for (normed, &s_val) in s_mid_normed.iter_mut().zip(s_mid.iter()) {
                *normed = s_val * inv_rms_ffn;
            }

            for f in 0..ffn_dim {
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for (d, &m_val) in s_mid_normed.iter().enumerate() {
                    gate += m_val * subnode.w_gate[d * ffn_dim + f];
                    up += m_val * subnode.w_up[d * ffn_dim + f];
                }

                let sig = 1.0 / (1.0 + (-gate).exp());
                let swish = gate * sig;
                let ffn_val = swish * up;

                for (d, t_val) in temp_out.iter_mut().enumerate() {
                    *t_val += ffn_val * subnode.w_down[f * d_head + d];
                }
            }

            for (out_val, (temp_val, s_val)) in self
                .p_out_buf
                .iter_mut()
                .zip(temp_out.iter().zip(s_mid.iter()))
            {
                *out_val = s_val + alpha * temp_val;
            }
        }

        self.last_prediction.copy_from_slice(&self.p_out_buf);
        self.last_token_id = Some(token_id);

        let active_alpha = self.subnodes[active].alpha;
        let scale_factor = 1.0 / (1.0 + active_alpha * active_alpha).sqrt();
        let mut delta_n = vec![0.0f32; d_head];
        for (delta, (out_val, in_val)) in delta_n
            .iter_mut()
            .zip(self.p_out_buf.iter().zip(self.p_in_buf.iter()))
        {
            *delta = out_val - in_val;
        }

        let mut total_batch_credit = 0.0f32;
        let mut valid_credit_count = 0.0f32;

        for p_ref in particles.iter_mut() {
            let mut fw_agreement = 0.0;
            let sq_sum_p: f32 = p_ref.payload.iter().map(|&x| x * x).sum();
            let inv_rms_p = 1.0 / (sq_sum_p / (d_head as f32) + 1e-8).sqrt();
            for r in 0..d_head {
                let mut sum = 0.0;
                for c in 0..d_head {
                    sum += self.fast_weight[r * d_head + c] * (p_ref.payload[c] * inv_rms_p);
                }
                fw_agreement += sum * (p_ref.payload[r] * inv_rms_p);
            }
            let agreement_before = fw_agreement;

            let previous_credit = p_ref.credit;
            let previous_credit_valid = p_ref.credit_valid;

            for (payload_val, delta_val) in p_ref.payload.iter_mut().zip(delta_n.iter()) {
                *payload_val = (*payload_val + delta_val) * scale_factor;
            }
            p_ref.credit_valid = false;

            let mut fw_agreement_after = 0.0;
            let sq_sum_p_after: f32 = p_ref.payload.iter().map(|&x| x * x).sum();
            let inv_rms_p_after = 1.0 / (sq_sum_p_after / (d_head as f32) + 1e-8).sqrt();
            for r in 0..d_head {
                let mut sum = 0.0;
                for c in 0..d_head {
                    sum += self.fast_weight[r * d_head + c] * (p_ref.payload[c] * inv_rms_p_after);
                }
                fw_agreement_after += sum * (p_ref.payload[r] * inv_rms_p_after);
            }
            let agreement_after = fw_agreement_after;

            p_ref.credit = agreement_after - agreement_before;
            p_ref.credit_valid = true;
            total_batch_credit += p_ref.credit;
            valid_credit_count += 1.0;
            self.sum_credit_volatility += p_ref.credit.abs();

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

        if valid_credit_count > 0.0 {
            let mean_credit = total_batch_credit / valid_credit_count;
            self.subnodes[active].credit_stats.observe(mean_credit);
            if mean_credit > 0.0 {
                self.subnodes[active].health += mean_credit;
            } else {
                self.subnodes[active].health += mean_credit * 0.5; // lose health faster
            }
        }

        self.activation_count += batch_size as u64;
        self.recent_activation_count += batch_size as u64;
        self.cumulative_sequence_len += batch_size as u64;
        self.subnodes[active].activation_count += batch_size as u64;

        let mut fw_sq_sum = 0.0f32;
        for &w in self.fast_weight.iter() {
            fw_sq_sum += w * w;
        }
        let fw_density = (fw_sq_sum / (d_head * d_head) as f32).sqrt();
        self.sum_memory_density += fw_density * batch_size as f32;
        self.attention_ops_count += batch_size as u64;

        self.try_subnode_neurogenesis();
        self.prune_dominated_subnodes();
    }

    pub fn extract_and_reset_metrics(&mut self) -> NodeMetrics {
        let metrics = NodeMetrics {
            sum_hop_count: self.sum_hop_count,
            halted_particles_count: self.halted_particles_count,
            sum_squared_energy: self.sum_squared_energy,
            total_particles_processed: self.total_particles_processed,
            sum_memory_density: self.sum_memory_density,
            attention_ops_count: self.attention_ops_count,
            sum_credit_volatility: self.sum_credit_volatility,
            sum_temporal_affinity: self.sum_temporal_affinity,
            active_subnodes_count: self.subnodes.len() as u64,
        };

        self.sum_hop_count = 0;
        self.halted_particles_count = 0;
        self.sum_squared_energy = 0.0;
        self.total_particles_processed = 0;
        self.sum_memory_density = 0.0;
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
    pub sum_memory_density: f32,
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
        let mut node = MicroBlockNode::new(0, config, false);

        let p1 = Particle::new(ParticleHeader::new(0, 0, 1.0), vec![0.5f32; 64]);
        let p2 = Particle::new(ParticleHeader::new(1, 0, 1.0), vec![0.8f32; 64]);
        let mut batch = vec![p1, p2];

        node.process_batch(&mut batch, Some(0.01));
        assert_eq!(node.activation_count, 2);
        assert_eq!(node.cumulative_sequence_len, 2);
    }
}
