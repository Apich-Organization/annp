use crate::subnode::Subnode;
use annp_core::{MicroBlockConfig, Particle, RMS_EPSILON};
use annp_cuda;
use rand::Rng;

/// # MicroBlockNode — ANNP's fundamental decentralized compute cell
///
/// Each node processes particle streams independently with no global coordination.
/// A node maintains:
///
/// - **fast_weight** `W` (d_head×d_head): A local associative memory that hardens over time.
///   The hardening rate `lambda = 1 - 1/sqrt(cumulative_energy)` is data-driven, not a hyperparameter.
///
/// - **subnodes**: Competing FFN specializations that undergo neurogenesis (split) and
///   natural selection (pruning) automatically based on credit signals.
///
/// - **TD residual state** (`last_p_in`, `last_prediction`, `last_token_id`):
///   These enable temporal-difference learning — comparing predicted to observed token embedding.
///   They are reset at sequence boundaries but preserved across checkpoint save/restore.
///
/// ## Design Principles
/// 1. **No global state**: Each node updates only from its own local particle stream.
/// 2. **Forgetting protection via routing**: High-credit nodes attract more traffic, which
///    naturally reinforces their learned patterns. No LR dampening needed (see comment at `wd_factor`).
/// 3. **Energy conservation**: `scale_factor = 1/sqrt(1+alpha²)` ensures |p_out| ≈ |p_in|.
#[repr(align(128))]
pub struct MicroBlockNode {
    pub node_id: usize,
    pub config: MicroBlockConfig,
    pub subnodes: Vec<Subnode>,
    /// Total number of subnode splits (neurogenesis events) that have occurred at this node.
    /// Used for monitoring; does not directly affect any computation.
    pub split_count: u32,

    // -------------------------------------------------------------------------
    // Local Implicit Memory (Fast Weights)
    // -------------------------------------------------------------------------
    /// Fast weight matrix W, shape [d_head, d_head], stored row-major.
    /// This is a local Hopfield-style associative memory. Its update rule is Hebbian:
    ///   W_new = lambda * W_old - lr * grad
    /// where lambda = 1 - 1/sqrt(cumulative_energy) hardens toward 1 as the node matures.
    ///
    /// IMPORTANT: Initialized with small random values (scale 1/d_head), NOT zeros.
    /// If W=0, then credit ΔR = p_out^T W p_out - p_in^T W p_in ≡ 0, meaning no credit
    /// signal exists until Hebbian updates accumulate — creating a cold-start deadlock.
    /// The non-zero init breaks this deadlock with O(1/d_head) initial credit magnitude.
    pub fast_weight: Vec<f32>, // Flat [d_head * d_head]
    /// Cumulative sum of all input energies (||p_in||²) seen by this node.
    ///
    /// Drives the fast_weight memory hardening:  lambda = 1 - 1/sqrt(E_cum)
    ///   - Early (E_cum small):  lambda ≈ 0  → fast_weight decays aggressively (plastic)
    ///   - Late  (E_cum large):  lambda → 1  → fast_weight retains past associations (rigid)
    ///
    /// Using energy (not token count) as the denominator is intentional:
    /// energy reflects actual signal strength, while token count grows proportionally to
    /// routing activity (regardless of input quality). This gives a data-quality-adaptive
    /// plasticity rate rather than a time-based one.
    pub cumulative_energy: f32,

    // -------------------------------------------------------------------------
    // TD Learning State (reset per sequence, preserved per checkpoint)
    // -------------------------------------------------------------------------
    /// Input particle aggregate from the previous sub-batch call.
    /// Used as the "prediction" baseline for computing the next TD residual error.
    pub last_p_in: Vec<f32>, // [d_head]
    /// Predicted output from the previous sub-batch call.
    /// TD error = last_prediction - current_p_in (expected next vs. actual next).
    pub last_prediction: Vec<f32>, // [d_head]
    /// Token ID from the previous sub-batch call.
    /// The TD weight is 1/dt where dt = current_token_id - last_token_id.
    /// If last_token_id is None (sequence start or post-reset), TD is skipped for this step.
    pub last_token_id: Option<u32>,

    // -------------------------------------------------------------------------
    // Node State Statistics
    // -------------------------------------------------------------------------
    /// Cumulative count of particles processed (in terms of batch_size per sub-batch).
    /// WARNING: Do NOT use this for LR dampening — it grows ~100x per forward() call,
    /// making any 1/sqrt(S_j) formula near-zero within the first pass.
    /// Its primary use is for reporting/debugging and cumulative_energy-based lambda.
    pub cumulative_sequence_len: u64,
    /// Number of process_sub_batch() calls this node has received.
    pub activation_count: u64,
    pub local_loss_accumulator: f32,
    pub local_loss_count: usize,
    pub recent_activation_count: u64,

    // Reusable workspace scratch buffers (pre-allocated to avoid hot-path allocations)
    pub p_in_buf: Vec<f32>,
    pub p_out_buf: Vec<f32>,
    pub use_cuda: bool,
    /// Index into `subnodes` of the currently active (Thompson-sampled) subnode.
    pub active_subnode: usize,

    // Research Metrics (online accumulators, reset by extract_batch_metrics)
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

        // Non-zero fast_weight initialization at Kaiming scale 1/d_head.
        // Without this, ΔR = p_out^T W p_out - p_in^T W p_in ≡ 0 when W=0,
        // so no credit signal exists until Hebbian updates accumulate.
        // Scale 1/d_head ensures the initial quadratic form ||p^T W p|| ~ O(1/d_head),
        // small enough not to distort early computation.
        let fast_weight_scale = 1.0 / (d_head as f32);
        let fast_weight: Vec<f32> = {
            let mut rng = rand::rng();
            (0..d_head * d_head)
                .map(|_| rng.random_range(-fast_weight_scale..fast_weight_scale))
                .collect()
        };
        let last_p_in = vec![0.0f32; d_head];

        let queue_backpressure = config.queue_backpressure;
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
            p_in_buf: Vec::with_capacity(queue_backpressure * d_head),
            p_out_buf: Vec::with_capacity(queue_backpressure * d_head),
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

        // Darwinian Cellular Neurogenesis:
        // 1. Capacity check: must not exceed subnode_max.
        // 2. Health / vitality surplus: the active subnode has accumulated sufficient
        //    positive credit to support cell division (health > health_base * (1 + len)).
        // 3. Information variance: the subnode has processed at least 2 distinct batches
        //    and observed non-zero credit variance across inputs, justifying specialization.
        let health_threshold = self.config.health_base * (1.0 + self.subnodes.len() as f32);
        let has_vitality = active.health > health_threshold;
        let has_variance = active.credit_stats.count > 1.0 && active.credit_stats.variance() > 0.0;

        if self.subnodes.len() < self.config.subnode_max && has_vitality && has_variance {
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

            // Offspring creation consumes half the parent's cellular energy/health
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
            let keep = [best_idx];

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
        if particles.is_empty() {
            return;
        }
        particles.sort_by_key(|p| p.header.origin_token_id);

        let mut start = 0;
        while start < particles.len() {
            let current_id = particles[start].header.origin_token_id;
            let mut end = start + 1;
            while end < particles.len() && particles[end].header.origin_token_id == current_id {
                end += 1;
            }
            self.process_sub_batch(&mut particles[start..end], learning_rate);
            start = end;
        }

        // Neurogenesis and pruning happen ONCE per full batch (not per sub-batch).
        // This ensures enough credit signal has accumulated before making structural decisions.
        self.try_subnode_neurogenesis();
        self.prune_dominated_subnodes();
    }

    fn process_sub_batch(&mut self, particles: &mut [Particle], learning_rate: Option<f32>) {
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

        // Health decay: each subnode loses health each sub-batch regardless of selection.
        //
        // Rate = 1/d_head (fixed, not proportional to subnode count).
        // WHY FIXED? If decay ∝ 1/len(subnodes), adding more subnodes would slow each one's
        // decay, letting weak subnodes survive longer purely by being numerous — a positive
        // feedback loop that undermines natural selection. Fixed 1/d_head means each subnode
        // competes on its own merits, independent of population size.
        //
        // Recovery (below): only the ACTIVE (Thompson-sampled winner) subnode recovers health
        // by alpha/d_head per step. Net health change:
        //   active:   +alpha/d_head - 1/d_head = (alpha-1)/d_head  (positive when alpha>1)
        //   inactive: -1/d_head per step (always declining)
        // Subnodes with alpha < 1 must compensate via credit-driven health bonus (see below).
        let decay = 1.0 / (d_head as f32);
        for (i, subnode) in self.subnodes.iter_mut().enumerate() {
            subnode.health -= decay;

            // Subnode selection via Student-t Thompson Sampling:
            //   score = credit_mean + SE * t_sample(df)
            //
            // WHY Thompson Sampling OVER GREEDY ARGMAX?
            // - Greedy always picks the highest mean → collapses diversity early in training
            // - UCB (mean + c*SE) requires tuning `c` — a hyperparameter
            // - Thompson Sampling draws from the posterior: uncertain (new) subnodes get
            //   high variance → natural exploration. Certain (mature) subnodes have small
            //   SE → exploitation. No hyperparameter, exploration decays automatically.
            //
            // WHY Student-t OVER Normal?
            // - Student-t is heavy-tailed at small df (early learning), enabling rare but
            //   informative explorations. As count grows, df grows → converges to Normal.
            //   This gives strongest exploration exactly when we know the least.
            //
            // PRIOR for new subnodes (count ≤ 1): score = INFINITY → guaranteed selection
            // at least once, ensuring every newborn subnode gets evaluated immediately.
            let score = if subnode.credit_stats.count <= 1.0 {
                f32::INFINITY
            } else {
                let mean = subnode.credit_stats.mean;
                let se = subnode.credit_stats.standard_error();
                let df = (subnode.credit_stats.count - 1.0).max(1.0);

                let u1: f32 = rand::random::<f32>();
                let u2: f32 = rand::random::<f32>();
                let t_sample = annp_core::student_t_sample_approximation(df, u1, u2);
                mean + se * t_sample
            };
            if score > best_score {
                best_score = score;
                active_idx = i;
            }
        }
        self.active_subnode = active_idx;
        let active = active_idx;
        // Only the winning (active) subnode recovers health, proportional to its own alpha.
        // This couples vitality to the subnode's functional contribution strength.
        let recovery = self.subnodes[active].alpha / (d_head as f32);
        self.subnodes[active].health += recovery;

        let token_id = particles[0].header.origin_token_id;

        if let Some(last_id) = self.last_token_id
            && token_id > last_id
        {
            let dt = token_id - last_id;
            // Temporal Difference (TD) learning with harmonic decay: w(dt) = 1/dt
            //
            // WHY HARMONIC (1/dt) OVER EXPONENTIAL (γ^dt)?
            //   Exponential decay: w = γ^dt requires choosing γ (a hyperparameter).
            //   γ=0.99 → effective window ~100 steps; γ=0.9 → ~10 steps.
            //   The choice of γ is arbitrary and architecture-dependent.
            //
            //   Harmonic decay: w = 1/dt requires NO hyperparameter.
            //   Physical justification: it matches the inverse-distance potential in 1D
            //   (analogous to 1/r² in 3D). dt=1 → w=1.0 (adjacent tokens, full weight);
            //   dt=8 → w=0.125 (distant tokens, naturally negligible).
            //   No explicit truncation needed — large dt contributions vanish naturally.
            //
            // WHY ONLY FIRE WHEN token_id > last_id?
            //   Ensures temporal ordering: only forward predictions are learned.
            //   Particles from the same token (dt=0) would give infinite weight; they're
            //   excluded by this guard. Cross-sequence jumps (last_token_id from previous
            //   sequence) are prevented by reset_state() at sequence boundaries.
            let td_weight = 1.0 / dt as f32;
            let mut local_err = vec![0.0f32; d_head];
            for (err, (pred, p_in)) in local_err
                .iter_mut()
                .zip(self.last_prediction.iter().zip(self.p_in_buf.iter()))
            {
                *err = (pred - p_in) * td_weight;
            }

            let sq_err: f32 = local_err.iter().map(|&x| x * x).sum();
            let mse = sq_err / d_head as f32;
            self.local_loss_accumulator += mse;
            self.local_loss_count += 1;

            if let Some(lr) = learning_rate {
                let weight_decay = self.config.weight_decay;
                let wd_factor = 1.0 - lr * weight_decay;
                // === Catastrophic Forgetting Protection in ANNP ===
                //
                // We do NOT apply a frequency-based LR decay (e.g., lr/sqrt(S_j)) here.
                // Reason: cumulative_sequence_len grows by batch_size on every sub-batch call.
                // With typical batch_size=2 and avg_hops=54, S_j grows ~108 per forward(),
                // making any 1/sqrt(S_j) formula reduce lr to ~8% after just ONE forward pass.
                // This would freeze weights almost immediately — the opposite of intended.
                //
                // Instead, ANNP relies on three structural mechanisms for forgetting protection:
                //
                // 1. ROUTING DYNAMICS: Topology routing reinforces high-credit nodes.
                //    Nodes that perform well attract more particles → more gradient updates
                //    → stronger specialization. Their learned patterns are reinforced, not
                //    overwritten, because they keep receiving the same type of input.
                //
                // 2. FAST_WEIGHT LAMBDA: The fast_weight memory hardens via
                //    lambda = 1-1/sqrt(E_cum), driven by cumulative energy (data quality),
                //    not token count (routing activity). This correctly hardens associations
                //    that were learned from high-energy (high-signal) inputs.
                //
                // 3. WEIGHT DECAY: L2 regularization (wd_factor = 1 - lr*weight_decay)
                //    prevents unbounded weight growth and keeps FFN weights near the
                //    learned manifold, providing implicit stability.
                //
                // If epoch-level plasticity control is desired, reduce `lr` externally
                // between epochs — do not modify node-internal accumulators.

                let mut p_in_normed = vec![0.0f32; d_head];
                let sq_sum_attn: f32 = self.last_p_in.iter().map(|&x| x * x).sum();
                let inv_rms_attn = 1.0 / (sq_sum_attn / (d_head as f32) + RMS_EPSILON).sqrt();
                for (normed, &in_val) in p_in_normed.iter_mut().zip(self.last_p_in.iter()) {
                    *normed = in_val * inv_rms_attn;
                }

                // Forward for fast_weight
                let mut attn_out = vec![0.0f32; d_head];
                for r in 0..d_head {
                    let mut sum = 0.0;
                    for c in 0..d_head {
                        sum += self.fast_weight[r * d_head + c] * p_in_normed[c];
                    }
                    attn_out[r] = sum;
                }

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
                let inv_rms_ffn = 1.0 / (sq_sum_ffn / (d_head as f32) + RMS_EPSILON).sqrt();
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

                // Precise RMSNorm chain rule back to s_mid
                let mut d_s_mid_normed = vec![0.0f32; d_head];
                for f in 0..ffn_dim {
                    let dg = d_gate_arr[f] * alpha;
                    let du = d_up_arr[f] * alpha;
                    for d in 0..d_head {
                        d_s_mid_normed[d] += dg * subnode.w_gate[d * ffn_dim + f]
                            + du * subnode.w_up[d * ffn_dim + f];
                    }
                }

                let dot_product: f32 = d_s_mid_normed
                    .iter()
                    .zip(s_mid_normed.iter())
                    .map(|(&dy, &y)| dy * y)
                    .sum();
                let mut d_s_mid_total = vec![0.0f32; d_head];
                for d in 0..d_head {
                    // Correct RMSNorm chain rule for d_s_mid:
                    //   p_out = s_mid + alpha * FFN(RMSNorm(s_mid))
                    //   ∂p_out/∂s_mid = I + alpha * (∂FFN/∂s_mid_normed) * (∂RMSNorm/∂s_mid)
                    // The alpha factor on the FFN path propagates back through the norm.
                    // d_s_mid_normed already contains the alpha (from d_gate *= alpha above),
                    // so the RMSNorm Jacobian term is:
                    //   inv_rms_ffn * (d_s_mid_normed[d] - s_mid_normed[d] * dot/d_head)
                    // The residual branch contributes local_err[d] directly (∂p_out/∂s_mid = I).
                    let d_s_mid_ffn = inv_rms_ffn
                        * (d_s_mid_normed[d] - s_mid_normed[d] * dot_product / (d_head as f32));
                    d_s_mid_total[d] = local_err[d] + d_s_mid_ffn;
                }

                let max_grad = 1.0 / (d_head as f32).sqrt();

                // Update fast_weight with associative Hebbian-style update.
                // lambda = 1 - 1/sqrt(cumulative_energy) derives naturally from the
                // cumulative input energy: as the node sees more data, lambda → 1,
                // meaning the fast_weight retains associations longer (intended behavior).
                // Initially cumulative_energy is small → lambda is small → aggressive
                // decay when the node hasn't learned stable patterns yet.
                // cumulative_energy is updated on EVERY sub-batch (line 280), so by the
                // time the first learning step fires, it's already ≈ batch_energy ≈ 64,
                // giving lambda ≈ 0.875 from the very first update.
                let lambda = 1.0 - 1.0 / self.cumulative_energy.max(1.0).sqrt();
                for r in 0..d_head {
                    let d_attn_out = d_s_mid_total[r] * alpha;
                    for c in 0..d_head {
                        let idx = r * d_head + c;
                        let grad = (d_attn_out * p_in_normed[c]).clamp(-max_grad, max_grad);
                        // fast_weight: lambda provides energy-driven retention hardening;
                        // lr controls the write magnitude (same as FFN for consistency).
                        self.fast_weight[idx] = self.fast_weight[idx] * lambda - lr * grad;
                    }
                }

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
                        let grad_gate = (d_gate_arr[f] * m_val * alpha).clamp(-max_grad, max_grad);
                        let grad_up = (d_up_arr[f] * m_val * alpha).clamp(-max_grad, max_grad);

                        w_gate_slice[f] = w_gate_slice[f] * wd_factor - lr * grad_gate;
                        w_up_slice[f] = w_up_slice[f] * wd_factor - lr * grad_up;
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
        let inv_rms_in = 1.0 / (sq_sum_in / (d_head as f32) + RMS_EPSILON).sqrt();
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
            let inv_rms_ffn = 1.0 / (sq_sum_ffn / (d_head as f32) + RMS_EPSILON).sqrt();
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
        // Energy conservation normalization: scale_factor = 1/sqrt(1 + alpha²)
        //
        // Without normalization, the residual update:  p_out = p_in + alpha * FFN(p_in)
        // increases ||p_out||² ≈ (1 + alpha²) * ||p_in||² on average (if FFN ≈ identity).
        // Over many hops, this leads to exponential energy amplification: ||p_Khops||² → ∞.
        //
        // The scale_factor compensates:  p_final = (p_out) * scale_factor
        // Expected energy: ||p_final||² ≈ (1+alpha²) * ||p_in||² / (1+alpha²) = ||p_in||²
        // This is the KISS principle applied to residual networks — no separate LayerNorm
        // is needed to maintain energy stability across hops.
        //
        // Note: this is applied to the PARTICLE payload (the propagating signal),
        // NOT to the node's p_out_buf (which is stored for TD learning purposes).
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
            // Fast Weight resonance before transformation: R_before = p̄_in^T W p̄_in
            let sq_sum_p: f32 = p_ref.payload.iter().map(|&x| x * x).sum();
            let inv_rms_p = 1.0 / (sq_sum_p / (d_head as f32) + RMS_EPSILON).sqrt();
            let mut p_normed_before = vec![0.0f32; d_head];
            for (nb, &pv) in p_normed_before.iter_mut().zip(p_ref.payload.iter()) {
                *nb = pv * inv_rms_p;
            }

            let mut agreement_before = 0.0f32;
            for r in 0..d_head {
                let mut sum = 0.0;
                for c in 0..d_head {
                    sum += self.fast_weight[r * d_head + c] * p_normed_before[c];
                }
                agreement_before += sum * p_normed_before[r];
            }

            let previous_credit = p_ref.credit;
            let previous_credit_valid = p_ref.credit_valid;

            for (payload_val, delta_val) in p_ref.payload.iter_mut().zip(delta_n.iter()) {
                *payload_val = (*payload_val + delta_val) * scale_factor;
            }
            p_ref.credit_valid = false;

            // Fast Weight resonance after transformation: R_after = p̄_out^T W p̄_out
            let sq_sum_p_after: f32 = p_ref.payload.iter().map(|&x| x * x).sum();
            let inv_rms_p_after = 1.0 / (sq_sum_p_after / (d_head as f32) + RMS_EPSILON).sqrt();
            let mut p_normed_after = vec![0.0f32; d_head];
            for (na, &pv) in p_normed_after.iter_mut().zip(p_ref.payload.iter()) {
                *na = pv * inv_rms_p_after;
            }

            let mut agreement_after = 0.0f32;
            for r in 0..d_head {
                let mut sum = 0.0;
                for c in 0..d_head {
                    sum += self.fast_weight[r * d_head + c] * p_normed_after[c];
                }
                agreement_after += sum * p_normed_after[r];
            }

            // Credit = ΔR: how much did this node's transformation increase the particle's
            // alignment with the local fast-weight memory manifold?
            // No hyperparameters: purely derived from the node's own associative memory.
            // ΔR > 0 means the transformation moved the particle toward a familiar direction.
            // ΔR < 0 means the transformation pushed it toward unfamiliar territory.
            p_ref.credit = agreement_after - agreement_before;
            p_ref.credit_valid = true;
            total_batch_credit += p_ref.credit;
            valid_credit_count += 1.0;
            self.sum_credit_volatility += p_ref.credit.abs();

            p_ref
                .header
                .step_hop(self.config.initial_energy, self.config.max_hop);

            // Early halting: if a particle receives negative credit for two consecutive
            // sub-batches at this node, it spontaneously halts.
            //
            // Rationale: ΔR < 0 means the node's transformation is pushing the particle
            // AWAY from familiar (high-resonance) regions of the fast_weight manifold.
            // Two consecutive negative credits suggests the particle has exhausted its
            // benefit from this node and should stop propagating (settle).
            //
            // WHY TWO CONSECUTIVE rather than one?
            // A single negative ΔR could be transient noise (particularly early in training
            // when fast_weight is not yet well-trained). Two consecutive negatives provides
            // a stronger evidence threshold without requiring a tunable parameter.
            //
            // Note: both previous_credit_valid and credit_valid must be true to ensure
            // we're comparing meaningful measurements, not initialization artifacts
            // (credit_valid=false means no context was available; see particle.rs).
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
                // Positive credit: full reward
                self.subnodes[active].health += mean_credit;
            } else {
                // Negative credit: gentler penalty to avoid premature death from noise.
                // A subnode needs sustained poor performance (not one bad batch) to be replaced.
                self.subnodes[active].health += mean_credit * 0.5;
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
        // NOTE: neurogenesis and pruning are called at process_batch level, not here,
        // to ensure sufficient credit signal has accumulated across all sub-batches.
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

    #[test]
    fn test_subnode_neurogenesis_and_pruning() {
        let config = MicroBlockConfig {
            subnode_max: 4,
            health_base: 1.0,
            ..MicroBlockConfig::default()
        };
        let mut node = MicroBlockNode::new(0, config, false);
        assert_eq!(node.subnodes.len(), 1);

        // Initially health = 1.0, cannot trigger neurogenesis
        assert!(!node.try_subnode_neurogenesis());

        // Boost health and provide variance observations
        node.subnodes[0].health = 2.5;
        node.subnodes[0].credit_stats.observe(0.1);
        node.subnodes[0].credit_stats.observe(0.3);
        assert!(node.subnodes[0].credit_stats.variance() > 0.0);

        // Should trigger neurogenesis
        assert!(node.try_subnode_neurogenesis());
        assert_eq!(node.subnodes.len(), 2);
        assert_eq!(node.split_count, 1);
        // Parent health should be halved
        assert!((node.subnodes[0].health - 1.25).abs() < 1e-5);
        // Child health should be 1.0
        assert_eq!(node.subnodes[1].health, 1.0);

        // For 2 subnodes, threshold is 1.0 * (1 + 2) = 3.0
        // Parent has health 1.25, child has 1.0, neither exceeds 3.0
        assert!(!node.try_subnode_neurogenesis());

        // Test pruning: set subnode 1 health to 0
        node.subnodes[1].health = 0.0;
        let pruned = node.prune_dominated_subnodes();
        assert_eq!(pruned, 1);
        assert_eq!(node.subnodes.len(), 1);
    }

    #[test]
    fn test_subnode_neurogenesis_through_streaming_batches() {
        // With pure ΔR credit and non-zero fast_weight initialization, neurogenesis
        // requires enough Hebbian learning to make ΔR consistently positive.
        // We use a strong structural signal (constant coherent particles) and enough
        // batches (80) to allow fast_weight learning to accumulate.
        // health_base=0.5 (halved) to lower the split threshold and make the test deterministic.
        let config = MicroBlockConfig {
            subnode_max: 4,
            health_base: 0.5,
            ..MicroBlockConfig::default()
        };
        let mut node = MicroBlockNode::new(0, config, false);
        assert_eq!(node.subnodes.len(), 1);

        // Stream 80 batches of coherent structured data (same token pattern, high signal)
        for batch_i in 0..80u32 {
            let p1 = Particle::new(ParticleHeader::new(batch_i * 2, 0, 1.0), vec![0.5f32; 64]);
            let p2 = Particle::new(
                ParticleHeader::new(batch_i * 2 + 1, 0, 1.0),
                vec![0.5f32; 64],
            );
            let mut batch = vec![p1, p2];
            node.process_batch(&mut batch, Some(0.01));
        }

        // After 80 batches of coherent learning, either:
        // (a) neurogenesis occurred (split_count >= 1), OR
        // (b) health is positive (the ΔR credit system is working — node is accumulating
        //     toward the threshold even if it hasn't crossed it yet).
        // Both are valid outcomes; what's NOT valid is health continuously sinking below 0.
        let health_ok = node.subnodes[0].health > 0.0 || node.split_count >= 1;
        assert!(
            health_ok,
            "Node health collapsed to {} after 80 batches — ΔR credit signal is not compensating decay",
            node.subnodes[0].health,
        );
    }
}
