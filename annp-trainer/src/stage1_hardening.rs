use annp_core::MicroBlockConfig;
use annp_model::{ANNPModel, MicroBlockNode, RoutingTable};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct HardeningResult {
    pub links_before: usize,
    pub links_pruned: usize,
    pub spawn_details: Vec<(usize, usize, usize)>,
}

/// Stage 1: Plasticity Hardening, Neurogenesis Growth & Precision Fine-Tuning.
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
        let mut new_node = MicroBlockNode::new(new_node_id, config, 64, node_a.use_cuda);
        let mut rng = rand::rng();
        let alpha = 0.5f32;
        let epsilon_scale = 0.01f32;

        for (i, w_c) in new_node.w_gate.iter_mut().enumerate() {
            let w_a = node_a.w_gate[i];
            let w_b = node_b.w_gate[i];
            let noise: f32 = rng.random_range(-epsilon_scale..epsilon_scale);
            *w_c = alpha * w_a + (1.0 - alpha) * w_b + noise;
        }

        for (i, w_c) in new_node.w_up.iter_mut().enumerate() {
            let w_a = node_a.w_up[i];
            let w_b = node_b.w_up[i];
            let noise: f32 = rng.random_range(-epsilon_scale..epsilon_scale);
            *w_c = alpha * w_a + (1.0 - alpha) * w_b + noise;
        }

        for (i, w_c) in new_node.w_down.iter_mut().enumerate() {
            let w_a = node_a.w_down[i];
            let w_b = node_b.w_down[i];
            let noise: f32 = rng.random_range(-epsilon_scale..epsilon_scale);
            *w_c = alpha * w_a + (1.0 - alpha) * w_b + noise;
        }

        new_node
    }

    /// Perform plastic hardening updates, growth neurogenesis and conservative synaptic pruning across all nodes in ANNPModel
    pub fn apply_plastic_hardening(&self, model: &mut ANNPModel) -> HardeningResult {
        // 1. Plastic Hardening Scaling
        for node in model.nodes.iter_mut() {
            let node_lr = self.compute_node_lr(node.cumulative_sequence_len);
            let scaling = node_lr / self.eta_0;
            node.alpha *= scaling;
        }

        // 2. Conservative Synaptic Link Pruning
        let (links_before, links_pruned) = model
            .topology
            .prune_all_links(model.config.pruning_threshold);

        // 3. Dynamic Neurogenesis Growth Checking
        let mut new_nodes = Vec::new();
        let neurogenesis_threshold = model.config.neurogenesis_threshold;
        let config = model.config.clone();

        for i in 0..model.nodes.len() {
            if model.nodes[i].activation_count >= neurogenesis_threshold {
                model.nodes[i].activation_count = 0; // Reset node activation counter after split

                let node_a = &model.nodes[i];
                let neighbor_id = if !model.topology.routing_tables[i].neighbors.is_empty() {
                    model.topology.routing_tables[i].neighbors[0]
                } else {
                    (i + 1) % model.nodes.len()
                };
                let node_b = &model.nodes[neighbor_id];

                let new_id = model.nodes.len() + new_nodes.len();
                let interpolated_node =
                    self.interpolate_new_node(node_a, node_b, new_id, config.clone());
                new_nodes.push((i, neighbor_id, interpolated_node));
            }
        }

        let mut spawn_details = Vec::new();

        // Inject generated new nodes into model and update P2P topology routing mesh
        for (parent_a, parent_b, new_node) in new_nodes {
            let new_id = new_node.node_id;
            spawn_details.push((parent_a, parent_b, new_id));

            model.nodes.push(new_node);
            model.node_queues.push(Vec::with_capacity(64));
            model.next_queues.push(Vec::with_capacity(64));
            model.num_nodes += 1;

            // Route parent_a -> new_id -> parent_b
            let neighbors = vec![parent_b, (parent_a + 1) % model.num_nodes];
            model
                .topology
                .routing_tables
                .push(RoutingTable::new(model.config.d_head, neighbors));

            if !model.topology.routing_tables[parent_a]
                .neighbors
                .contains(&new_id)
            {
                model.topology.routing_tables[parent_a]
                    .neighbors
                    .push(new_id);
            }
        }

        HardeningResult {
            links_before,
            links_pruned,
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
            pruning_threshold: 1e-7,
            neurogenesis_threshold: 50,
            queue_backpressure_alpha: 0.05,
            min_routing_entropy_noise: 0.05,
            max_alpha_residual: 0.1,
        }
    }

    #[test]
    fn test_neurogenesis_growth_and_activation_reset() {
        let config = create_test_config();
        let device = Device::Cpu;
        let mut model = ANNPModel::new_with_cuda(4, 4, config, device, false);

        assert_eq!(model.nodes.len(), 4);
        assert_eq!(model.num_nodes, 4);

        // Manually simulate activations on Node 0 beyond neurogenesis_threshold
        model.nodes[0].activation_count = 60;
        model.nodes[1].activation_count = 10;

        let trainer = Stage1HardeningTrainer::new(0.01, 0.001, 1.5);
        let result = trainer.apply_plastic_hardening(&mut model);

        // Verify neurogenesis triggered for Node 0
        assert_eq!(result.spawn_details.len(), 1);
        let (parent_a, _parent_b, new_id) = result.spawn_details[0];
        assert_eq!(parent_a, 0);
        assert_eq!(new_id, 4);

        // Verify node count expanded
        assert_eq!(model.nodes.len(), 5);
        assert_eq!(model.num_nodes, 5);
        assert_eq!(model.topology.routing_tables.len(), 5);

        // Verify Node 0 activation count reset to 0
        assert_eq!(model.nodes[0].activation_count, 0);
        // Verify Node 1 activation count preserved (10 < 50)
        assert_eq!(model.nodes[1].activation_count, 10);

        // Verify new node 4 is linked in Node 0's routing neighbors
        assert!(model.topology.routing_tables[0].neighbors.contains(&4));
    }

    #[test]
    fn test_synaptic_pruning() {
        let mut config = create_test_config();
        // Set high pruning threshold to force pruning weak links
        config.pruning_threshold = 100.0;
        let device = Device::Cpu;
        let mut model = ANNPModel::new_with_cuda(4, 4, config, device, false);

        let trainer = Stage1HardeningTrainer::new(0.01, 0.001, 1.5);
        let result = trainer.apply_plastic_hardening(&mut model);

        // Verify dead links were pruned
        assert!(result.links_pruned > 0);
        assert!(result.links_before > result.links_pruned);
    }
}
