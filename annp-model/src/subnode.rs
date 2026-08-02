use annp_core::OnlineStats;
use annp_cuda::ffi::CudaDeviceWeights;
use rand::Rng;

/// # Subnode — Competing FFN specialization within a MicroBlockNode
///
/// Each MicroBlockNode hosts 1..=subnode_max Subnodes. They compete via Thompson Sampling
/// (see `process_sub_batch` in `micro_block.rs`) for the right to process each sub-batch.
/// The winner earns health recovery; losers decay. When health reaches zero, the subnode
/// is pruned. When health exceeds the threshold, neurogenesis spawns a new sibling.
///
/// ## FFN Architecture
/// The Subnode implements a gated FFN (SwiGLU-style):
///   output = W_down @ (swish(W_gate @ x) ⊙ (W_up @ x))
/// where x = RMSNorm(p_in + alpha * fast_weight_output).
///
/// ## Key Design Invariants
/// - `alpha`: Residual scale. Initialized to 1/sqrt(d_head); inherited by children.
///   Controls how strongly this subnode modifies the particle vs. passing it through.
/// - `credit_stats`: Tracks ΔR observations for Thompson Sampling. NOT model weights —
///   it guides routing/selection but is reset on spawn (children start fresh).
/// - `health`: Viability score. Increases on selection (by alpha/d_head), decays each
///   step (by 1/d_head). Net positive only when alpha > 1 AND frequently selected.
#[derive(Debug)]
pub struct Subnode {
    pub subnode_id: usize,
    /// Gate projection matrix, shape [d_head, ffn_dim] stored row-major.
    pub w_gate: Vec<f32>,
    /// Up projection matrix, shape [d_head, ffn_dim] stored row-major.
    pub w_up: Vec<f32>,
    /// Down projection matrix, shape [ffn_dim, d_head] stored row-major.
    /// Scale initialized to scale/sqrt(d_head) to prevent output variance explosion:
    ///   Var(output) ≈ ffn_scale² × d_head × (1/d_head) = ffn_scale² (preserved).
    pub w_down: Vec<f32>,
    /// Residual connection scale factor. Initialized to 1/sqrt(d_head).
    /// Children inherit this value from their parent rather than re-randomizing,
    /// so they start specializing from the same functional scale as the parent.
    pub alpha: f32,
    pub activation_count: u64,
    /// Online statistics (EMA Welford) over observed ΔR credit values.
    /// Used exclusively for Thompson Sampling subnode selection — NOT model knowledge.
    /// `gamma` is set to 1 - 1/(d_head²) so the effective memory window ≈ d_head² steps,
    /// aligned with the fast_weight matrix size. This prevents credit statistics from
    /// becoming stale too quickly (γ too small) or forgetting important history (γ too large).
    pub credit_stats: OnlineStats,
    /// Darwinian cellular health. Drives neurogenesis (split when high) and pruning (die when ≤ 0).
    pub health: f32,
    /// Optional GPU weight mirror. None on CPU; populated by `sync_cuda_weights()`.
    pub d_weights: Option<CudaDeviceWeights>,
    /// Cumulative sum of input particle energies seen specifically by this subnode.
    /// Drives subnode-level fast_weight memory hardening: lambda = 1 - 1/sqrt(cumulative_energy).
    ///
    /// ### Mathematical Rationale:
    /// Node-level cumulative_energy reflects global container age. However, when a child subnode
    /// is spawned via neurogenesis, inheriting node-level cumulative_energy would assign it
    /// rigid memory retention (lambda ≈ 0.999), preventing the newborn subnode from rapidly
    /// differentiating away from parent weights. Resetting cumulative_energy to 0.0 on spawn
    /// ensures newborn subnodes start with lambda = 0 (full plasticity), enabling rapid manifold specialization.
    pub cumulative_energy: f32,

    /// Subnode-level TD prediction baseline from the previous activation of THIS specific subnode.
    ///
    /// ### Mathematical Rationale:
    /// Isolates temporal difference learning to individual functional specialists (subnodes).
    /// Prevents cross-talk corruption where heterogeneous particle streams visiting the same container
    /// node overwrite a single shared TD state vector.
    pub last_p_in: Vec<f32>,

    /// Subnode-level predicted output vector from the previous activation of THIS specific subnode.
    pub last_prediction: Vec<f32>,

    /// Subnode-level token ID from the previous activation of THIS specific subnode.
    /// Used to compute token gap dt = current_token_id - subnode.last_token_id and harmonic weight w(dt) = 1/dt.
    pub last_token_id: Option<u32>,
}

impl Clone for Subnode {
    fn clone(&self) -> Self {
        Self {
            subnode_id: self.subnode_id,
            w_gate: self.w_gate.clone(),
            w_up: self.w_up.clone(),
            w_down: self.w_down.clone(),
            alpha: self.alpha,
            activation_count: self.activation_count,
            credit_stats: self.credit_stats,
            health: self.health,
            d_weights: None, // Will be initialized by the model
            cumulative_energy: self.cumulative_energy,
            last_p_in: self.last_p_in.clone(),
            last_prediction: self.last_prediction.clone(),
            last_token_id: self.last_token_id,
        }
    }
}

impl Subnode {
    /// Creates a randomly initialized Subnode using He/Kaiming initialization.
    ///
    /// # Parameters
    /// - `gamma`: EMA decay for `credit_stats`. Should be `1 - 1/(d_head²)` (set by caller
    ///   in `MicroBlockNode::new`), NOT a fixed constant like 0.99. Rationale:
    ///   - Effective memory window N ≈ 1/(1-γ) = d_head² (e.g., 4096 for d_head=64)
    ///   - Aligns the credit statistics window with the fast_weight matrix size (d_head×d_head),
    ///     so credit history covers a comparable number of patterns as fast_weight can store.
    ///   - Fixed γ=0.99 would give N≈100, too short to distinguish stable subnode differences.
    pub fn new_random(
        subnode_id: usize,
        d_head: usize,
        ffn_dim: usize,
        alpha_init: f32,
        gamma: f32,
    ) -> Self {
        let mut rng = rand::rng();
        // He/Kaiming scale: sqrt(2 / (fan_in + fan_out))
        let scale = (2.0 / (d_head + ffn_dim) as f64).sqrt() as f32;

        let w_gate = (0..d_head * ffn_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        let w_up = (0..d_head * ffn_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();

        // w_down scale = scale / sqrt(d_head) — extra downstream normalization.
        //
        // Why smaller? The FFN output is a sum over ffn_dim terms:
        //   output[d] = Σ_f (swish(gate_f) * up_f) * w_down[f, d]
        // Variance of this sum ≈ ffn_dim * scale_gate² * scale_down².
        // Without the extra 1/sqrt(d_head) factor, output variance grows with ffn_dim.
        // Dividing w_down scale by sqrt(d_head) keeps Var(output) ≈ 1, preventing
        // the residual update from dominating p_in at initialization.
        // This is the standard "output-layer scaling" trick from weight initialization theory.
        let w_down_scale = scale / (d_head as f64).sqrt() as f32;
        let w_down = (0..ffn_dim * d_head)
            .map(|_| rng.random_range(-w_down_scale..w_down_scale))
            .collect();

        Self {
            subnode_id,
            w_gate,
            w_up,
            w_down,
            alpha: alpha_init,
            activation_count: 0,
            credit_stats: OnlineStats::new(gamma),
            health: 1.0,
            d_weights: None,
            cumulative_energy: 0.0,
            last_p_in: vec![0.0f32; d_head],
            last_prediction: vec![0.0f32; d_head],
            last_token_id: None,
        }
    }

    /// Spawns a new child Subnode from a parent, with a small data-driven perturbation.
    ///
    /// ## Perturbation Scale: epsilon = 1/sqrt(d_head * ffn_dim)
    ///
    /// WHY NOT FIXED 0.01?
    /// The parent's weight scale at init ≈ sqrt(2/(d_head+ffn_dim)).
    /// For d_head=64, ffn_dim=512: scale ≈ 0.055, epsilon ≈ 1/sqrt(32768) ≈ 0.0055.
    /// This gives perturbation ≈ 10% of initial weight magnitude — a consistent ratio
    /// regardless of model size. A hardcoded 0.01 would be 18% for large d and 200%
    /// for small d, causing either under- or over-diversification.
    ///
    /// ## Alpha Inheritance
    ///
    /// The child inherits `parent.alpha` rather than re-randomizing. Alpha controls
    /// the residual connection strength (how much this subnode modifies the particle).
    /// Starting from the parent's operating scale lets the child immediately function
    /// at a comparable contribution level and specialize from there via gradient updates.
    /// Re-randomizing alpha would require the child to re-discover the correct scale
    /// from scratch, slowing specialization.
    ///
    /// ## Bayesian Prior Inheritance for Thompson Sampling
    ///
    /// Rather than starting from fresh zero stats (`count=0`, which produces `m2=0 => SE=0`
    /// and paralyzes Student-t heavy-tailed exploration after the 2nd sample), the child
    /// inherits `parent.credit_stats` with `count` set to 2.0.
    ///
    /// ### Mathematical Rationale:
    /// Setting `count = 2.0` yields degrees of freedom `df = count - 1.0 = 1.0`, where Student-t
    /// reduces to a Cauchy distribution (maximum heavy tails). Inheriting the parent's non-zero
    /// variance ensures `SE = sqrt(m2 / count) > 0`, allowing the Cauchy distribution's heavy tails
    /// to generate natural, Bayesian-justified exploration opportunities while the child's
    /// plastic weights (lambda = 0) adapt to the new manifold. Zero extra hyperparameters.
    ///
    /// ## Initial Health — Cascade-Safety Invariant
    ///
    /// The child is initialized with `health = health_base` (from `MicroBlockConfig`),
    /// passed explicitly by the caller to avoid hardcoding.
    ///
    /// ### Why `health_base` and NOT a larger multiple?
    /// The Darwinian neurogenesis threshold for `n` existing subnodes is:
    ///   `threshold(n) = health_base * (1 + n)`
    /// After spawning, `n` increases by 1, so the child faces:
    ///   `threshold(n+1) = health_base * (n + 2)`
    /// For the first split (n=1 → child faces n+1=2 subnodes):
    ///   `threshold = health_base * (1 + 2) = 3 * health_base`
    /// The child's health must satisfy: `h_child_init < 3 * health_base`
    /// → `health_base < 3 * health_base` ✓ for any positive health_base.
    ///
    /// Any multiplier k ≥ 3 would violate this for the first split, causing
    /// the child (which inherits positive credit prior via Bayesian inheritance)
    /// to immediately trigger cascade neurogenesis upon first Thompson Sampling
    /// selection — filling `subnode_max` with untrained subnodes in a handful of batches.
    ///
    /// k=1 (`h_init = health_base`) is the largest cascade-safe integral multiplier
    /// for the first-split case. The newborn's plastic phase is protected against
    /// premature death by the λ-scaled credit penalty (λ=0 when E_cum=0 → zero
    /// death penalty on initial weight-perturbation noise).
    pub fn spawn_from_parent(
        subnode_id: usize,
        parent: &Subnode,
        d_head: usize,
        ffn_dim: usize,
        _gamma: f32,
        health_base: f32,
    ) -> Self {
        let mut rng = rand::rng();
        let epsilon = 1.0 / (d_head as f32 * ffn_dim as f32).sqrt();

        let w_gate = parent
            .w_gate
            .iter()
            .map(|&w| w + rng.random_range(-epsilon..epsilon))
            .collect();
        let w_up = parent
            .w_up
            .iter()
            .map(|&w| w + rng.random_range(-epsilon..epsilon))
            .collect();
        let w_down = parent
            .w_down
            .iter()
            .map(|&w| w + rng.random_range(-epsilon..epsilon))
            .collect();

        // Bayesian prior inheritance: inherit parent's empirical mean & variance,
        // but set count = 2.0 (df = 1.0 => Cauchy distribution heavy-tailed exploration)
        let mut child_credit_stats = parent.credit_stats;
        child_credit_stats.count = 2.0;

        Self {
            subnode_id,
            w_gate,
            w_up,
            w_down,
            alpha: parent.alpha, // Inherit operating scale
            activation_count: 0,
            credit_stats: child_credit_stats,
            // h_init = health_base (passed from MicroBlockConfig via caller).
            // Cascade-safety: health_base < 3*health_base (2-subnode threshold) for any
            // positive health_base. See fn doc for the full cascade-safety invariant proof.
            health: health_base,
            d_weights: None,
            cumulative_energy: 0.0, // Child starts plastic! (lambda = 0 → full plasticity)
            last_p_in: vec![0.0f32; d_head],
            last_prediction: vec![0.0f32; d_head],
            last_token_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_random_initialization() {
        let d_head = 64;
        let ffn_dim = 128;
        let alpha = 0.5;
        let gamma = 0.99;
        let node = Subnode::new_random(1, d_head, ffn_dim, alpha, gamma);

        assert_eq!(node.subnode_id, 1);
        assert_eq!(node.alpha, 0.5);
        assert_eq!(node.activation_count, 0);

        let w_len = d_head * ffn_dim;
        assert_eq!(node.w_gate.len(), w_len);
        assert_eq!(node.w_up.len(), w_len);
        assert_eq!(node.w_down.len(), w_len);
    }

    #[test]
    fn test_spawn_from_parent() {
        let d_head = 32;
        let ffn_dim = 64;
        let gamma = 0.99;
        let parent = Subnode::new_random(0, d_head, ffn_dim, 0.7, gamma);

        let child = Subnode::spawn_from_parent(1, &parent, d_head, ffn_dim, gamma, 1.0);

        assert_eq!(child.subnode_id, 1);
        assert_eq!(child.alpha, 0.7);

        let w_len = d_head * ffn_dim;
        assert_eq!(child.w_gate.len(), w_len);

        let mut w_gate_diff = 0.0;
        for i in 0..w_len {
            w_gate_diff += (child.w_gate[i] - parent.w_gate[i]).abs();
        }
        assert!(
            w_gate_diff > 0.0,
            "Child weights must be perturbed from parent"
        );

        let mut w_down_diff = 0.0;
        for i in 0..w_len {
            w_down_diff += (child.w_down[i] - parent.w_down[i]).abs();
        }
        assert!(
            w_down_diff > 0.0,
            "Child w_down must be perturbed from parent"
        );
    }
}
