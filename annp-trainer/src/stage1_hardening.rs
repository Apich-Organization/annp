use annp_model::ANNPModel;

#[derive(Debug, Clone)]
pub struct HardeningResult {
    pub links_before: usize,
    pub links_pruned: usize,
    pub spawn_details: Vec<(usize, usize, usize)>, // (node_id, old_subnode_count, new_subnode_count)
}

/// Stage 1: Plasticity Hardening, Subnode Neurogenesis Growth & Precision Fine-Tuning.
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

    /// Perform plastic hardening updates and subnode micro-column neurogenesis checking across all nodes
    pub fn apply_plastic_hardening(&self, model: &mut ANNPModel) -> HardeningResult {
        // 1. Plastic Hardening Scaling for primary subnodes (Direct assignment from alpha_init to prevent geometric compound decay)
        let alpha_init = model.config.alpha_init;
        for node in model.nodes.iter_mut() {
            let node_lr = self.compute_node_lr(node.cumulative_sequence_len);
            let scaling = node_lr / self.eta_0;
            if let Some(primary) = node.subnodes.first_mut() {
                primary.alpha = alpha_init * scaling;
            }
        }

        // 2. Subnode Micro-Column Neurogenesis Checking
        let mut spawn_details = Vec::new();

        for i in 0..model.nodes.len() {
            let node = &mut model.nodes[i];
            let count_before = node.subnodes.len();
            if node.try_subnode_neurogenesis() {
                let count_after = node.subnodes.len();
                spawn_details.push((i, count_before, count_after));
            }
        }

        let links_before = model
            .topology
            .routing_tables
            .iter()
            .map(|rt| rt.neighbors.len())
            .sum();

        HardeningResult {
            links_before,
            links_pruned: 0,
            spawn_details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_core::{MicroBlockConfig, NormStrategy};
    use candle_core::Device;

    fn create_test_config() -> MicroBlockConfig {
        MicroBlockConfig {
            num_shards: 4,
            mesh_rows: 2,
            mesh_cols: 2,
            d_head: 64,
            ffn_expansion: 8,
            initial_energy: 1.0,
            max_hop: 20,
            min_hop: 2,
            epsilon_p: 1e-4,
            epsilon_h: 0.05,
            temperature: 1.0,
            norm_strategy: NormStrategy::MicroRMSNorm,
            alpha_init: 0.01,
            sphere_radius: 1.0,
            lambda_temporal: 0.001,
            lambda_frequency: 0.01,
            eviction_threshold: 1e-4,
            neurogenesis_threshold: 50,
            subnode_max: 8,
            progressive_hardening_factor: 0.5,
            queue_backpressure_alpha: 0.05,
            min_routing_entropy_noise: 0.05,
            max_alpha_residual: 0.1,
        }
    }

    #[test]
    fn test_subnode_neurogenesis_growth_and_progressive_threshold() {
        let config = create_test_config();
        let device = Device::Cpu;
        let mut model = ANNPModel::new_with_cuda(4, 4, config, device, false);

        assert_eq!(model.nodes.len(), 4);
        assert_eq!(model.nodes[0].subnodes.len(), 1);
        assert_eq!(model.nodes[0].split_count, 0);

        // Manually simulate activations on Node 0 beyond base neurogenesis_threshold (50)
        model.nodes[0].activation_count = 60;
        model.nodes[1].activation_count = 10;

        let trainer = Stage1HardeningTrainer::new(0.01, 0.001, 1.5);
        let result = trainer.apply_plastic_hardening(&mut model);

        // Verify subnode neurogenesis triggered for Node 0
        assert_eq!(result.spawn_details.len(), 1);
        let (node_id, count_before, count_after) = result.spawn_details[0];
        assert_eq!(node_id, 0);
        assert_eq!(count_before, 1);
        assert_eq!(count_after, 2);

        // Verify global node count remained FIXED at 4
        assert_eq!(model.nodes.len(), 4);
        assert_eq!(model.nodes[0].subnodes.len(), 2);
        assert_eq!(model.nodes[0].split_count, 1);

        // Verify Node 0 activation count reset to 0 after subnode split
        assert_eq!(model.nodes[0].activation_count, 0);

        // Verify progressive threshold increased for split_count = 1 (50 * (1 + 0.5 * 1) = 75)
        let new_thresh = model.nodes[0]
            .config
            .current_neurogenesis_threshold(model.nodes[0].split_count);
        assert_eq!(new_thresh, 75);

        // 60 activations will NOT trigger split now since 60 < 75
        model.nodes[0].activation_count = 60;
        let result2 = trainer.apply_plastic_hardening(&mut model);
        assert_eq!(result2.spawn_details.len(), 0);

        // 80 activations WILL trigger split since 80 >= 75
        model.nodes[0].activation_count = 80;
        let result3 = trainer.apply_plastic_hardening(&mut model);
        assert_eq!(result3.spawn_details.len(), 1);
        assert_eq!(model.nodes[0].subnodes.len(), 3);
        assert_eq!(model.nodes[0].split_count, 2);
    }
}
