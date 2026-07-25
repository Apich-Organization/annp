use annp_core::Particle;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Q-Routing table maintained locally at each Micro-Block Node.
/// Size: [d_head, num_neighbors]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub d_head: usize,
    pub neighbors: Vec<usize>,
    pub weights: Vec<f32>, // Flat [d_head * num_neighbors]
}

impl RoutingTable {
    pub fn new(d_head: usize, neighbors: Vec<usize>) -> Self {
        let num_neighbors = neighbors.len();
        let mut rng = rand::thread_rng();
        let scale = (1.0 / (d_head as f32)).sqrt();

        let weights = (0..d_head * num_neighbors)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        Self {
            d_head,
            neighbors,
            weights,
        }
    }

    /// Predict next hop neighbor node index using Q-Routing dot product + Softmax
    pub fn select_next_hop(&self, particle: &Particle, temperature: f32) -> usize {
        let num_neighbors = self.neighbors.len();
        if num_neighbors == 0 {
            return 0;
        }

        let mut logits = vec![0.0f32; num_neighbors];
        for k in 0..num_neighbors {
            let mut dot = 0.0f32;
            for d in 0..self.d_head {
                dot += particle.payload[d] * self.weights[d * num_neighbors + k];
            }
            logits[k] = dot / temperature.max(1e-4);
        }

        // Softmax & Argmax sampling
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();

        let mut rng = rand::thread_rng();
        let p: f32 = rng.gen_range(0.0..1.0);
        let mut cum_sum = 0.0f32;

        for k in 0..num_neighbors {
            cum_sum += exps[k] / (sum_exp + 1e-8);
            if p <= cum_sum {
                return self.neighbors[k];
            }
        }

        self.neighbors[0]
    }
}

/// System P2P Topology Grid managing node connectivity and routing tables.
pub struct TopologyGrid {
    pub num_nodes: usize,
    pub routing_tables: Vec<RoutingTable>,
}

impl TopologyGrid {
    pub fn new(num_nodes: usize, d_head: usize, neighbors_per_node: usize) -> Self {
        let mut routing_tables = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            let mut neighbors = Vec::with_capacity(neighbors_per_node);
            for n in 1..=neighbors_per_node {
                let neighbor_id = (i + n * 7) % num_nodes; // Structured P2P mesh connections
                if neighbor_id != i {
                    neighbors.push(neighbor_id);
                }
            }
            routing_tables.push(RoutingTable::new(d_head, neighbors));
        }

        Self {
            num_nodes,
            routing_tables,
        }
    }
}
