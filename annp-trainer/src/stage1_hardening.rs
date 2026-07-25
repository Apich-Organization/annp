use annp_core::MicroBlockConfig;
use annp_model::{ANNPModel, MicroBlockNode};
use rand::Rng;

/// Stage 1: Plasticity Hardening & Precision Fine-Tuning.
pub struct Stage1HardeningTrainer {
    pub eta_0: f32,
    pub beta: f32,
    pub theta: f32,
}

impl Stage1HardeningTrainer {
    pub fn new(eta_0: f32, beta: f32, theta: f32) -> Self {
        Self { eta_0, beta, theta }
    }

    /// Compute node-specific plastic hardening learning rate: \eta_j = \eta_0 / (1 + \beta * S_j)^\theta
    pub fn compute_node_lr(&self, cumulative_seq_len: u64) -> f32 {
        let s_j = cumulative_seq_len as f32;
        self.eta_0 / (1.0 + self.beta * s_j).powf(self.theta)
    }

    /// Perform Midpoint Neurogenesis Interpolation: W_C = \alpha * W_A + (1-\alpha) * W_B + \epsilon
    pub fn interpolate_new_node(
        &self,
        node_a: &MicroBlockNode,
        node_b: &MicroBlockNode,
        new_node_id: usize,
        config: MicroBlockConfig,
    ) -> MicroBlockNode {
        let mut new_node = MicroBlockNode::new(new_node_id, config, 64);
        let mut rng = rand::thread_rng();
        let alpha = 0.5f32;
        let epsilon_scale = 0.01f32;

        for (i, w_c) in new_node.w_gate.iter_mut().enumerate() {
            let w_a = node_a.w_gate[i];
            let w_b = node_b.w_gate[i];
            let noise: f32 = rng.gen_range(-epsilon_scale..epsilon_scale);
            *w_c = alpha * w_a + (1.0 - alpha) * w_b + noise;
        }

        for (i, w_c) in new_node.w_up.iter_mut().enumerate() {
            let w_a = node_a.w_up[i];
            let w_b = node_b.w_up[i];
            let noise: f32 = rng.gen_range(-epsilon_scale..epsilon_scale);
            *w_c = alpha * w_a + (1.0 - alpha) * w_b + noise;
        }

        for (i, w_c) in new_node.w_down.iter_mut().enumerate() {
            let w_a = node_a.w_down[i];
            let w_b = node_b.w_down[i];
            let noise: f32 = rng.gen_range(-epsilon_scale..epsilon_scale);
            *w_c = alpha * w_a + (1.0 - alpha) * w_b + noise;
        }

        new_node
    }

    /// Perform plastic hardening updates across all nodes in ANNPModel
    pub fn apply_plastic_hardening(&self, model: &mut ANNPModel) {
        for node in model.nodes.iter_mut() {
            let node_lr = self.compute_node_lr(node.cumulative_sequence_len);
            let scaling = node_lr / self.eta_0;
            node.alpha *= scaling;
        }
    }
}
