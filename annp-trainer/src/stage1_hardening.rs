use annp_core::MicroBlockConfig;
use annp_model::{ANNPModel, MicroBlockNode, RoutingTable};
use rand::Rng;

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

    /// Perform plastic hardening updates, growth neurogenesis and conservative synaptic pruning across all nodes in ANNPModel
    pub fn apply_plastic_hardening(&self, model: &mut ANNPModel) {
        // 1. Plastic Hardening Scaling
        for node in model.nodes.iter_mut() {
            let node_lr = self.compute_node_lr(node.cumulative_sequence_len);
            let scaling = node_lr / self.eta_0;
            node.alpha *= scaling;
        }

        // 2. Conservative Synaptic Link Pruning
        model
            .topology
            .prune_all_links(model.config.pruning_threshold);

        // 3. Dynamic Neurogenesis Growth Checking
        let mut new_nodes = Vec::new();
        let neurogenesis_threshold = model.config.neurogenesis_threshold;
        let config = model.config.clone();

        for i in 0..model.nodes.len() {
            if model.nodes[i].activation_count >= neurogenesis_threshold {
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

        // Inject generated new nodes into model and update P2P topology routing mesh
        for (parent_a, parent_b, new_node) in new_nodes {
            let new_id = new_node.node_id;
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
    }
}
