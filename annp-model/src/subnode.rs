use annp_core::OnlineStats;
use annp_cuda::ffi::CudaDeviceWeights;
use rand::Rng;

#[derive(Debug)]
pub struct Subnode {
    pub subnode_id: usize,
    pub w_gate: Vec<f32>, // [d_head, ffn_dim]
    pub w_up: Vec<f32>,   // [d_head, ffn_dim]
    pub w_down: Vec<f32>, // [ffn_dim, d_head]
    pub alpha: f32,
    pub activation_count: u64,
    /// Empirical local credit; reset on checkpoint restore because it is an
    /// online decision aid, not model knowledge.
    pub credit_stats: OnlineStats,
    pub d_weights: Option<CudaDeviceWeights>,
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
            d_weights: None, // Will be initialized by the model
        }
    }
}

impl Subnode {
    pub fn new_random(subnode_id: usize, d_head: usize, ffn_dim: usize, alpha_init: f32, gamma: f32) -> Self {
        let mut rng = rand::rng();
        let scale = (2.0 / (d_head + ffn_dim) as f64).sqrt() as f32;

        let w_gate = (0..d_head * ffn_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();
        let w_up = (0..d_head * ffn_dim)
            .map(|_| rng.random_range(-scale..scale))
            .collect();

        let w_down_scale = scale * (1.0 / (d_head as f64).sqrt() as f32); // Mathematical variance preservation instead of arbitrary 0.05
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
            d_weights: None,
        }
    }

    /// Spawns a new subnode by inheriting parent subnode weights with a small orthogonal/mathematical perturbation
    pub fn spawn_from_parent(
        subnode_id: usize,
        parent: &Subnode,
        d_head: usize,
        ffn_dim: usize,
        gamma: f32,
    ) -> Self {
        let mut rng = rand::rng();
        // Base perturbation on standard deviations rather than fixed 0.01
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
        let w_down = vec![0.0f32; parent.w_down.len()];

        Self {
            subnode_id,
            w_gate,
            w_up,
            w_down,
            alpha: parent.alpha,
            activation_count: 0,
            credit_stats: OnlineStats::new(gamma),
            d_weights: None,
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

        let child = Subnode::spawn_from_parent(1, &parent, d_head, ffn_dim, gamma);

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

        assert!(child.w_down.iter().all(|&w| w == 0.0));
    }
}
