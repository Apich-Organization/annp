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
        let mut rng = rand::rng();
        let scale = (1.0 / (d_head as f32)).sqrt();

        let weights = (0..d_head * num_neighbors)
            .map(|_| rng.random_range(-scale..scale))
            .collect();

        Self {
            d_head,
            neighbors,
            weights,
        }
    }

    /// Add a new neighbor link to routing table, dynamically expanding weight matrix shape [d_head, num_neighbors]
    pub fn add_neighbor(&mut self, neighbor_id: usize) {
        if self.neighbors.contains(&neighbor_id) {
            return;
        }
        let old_num_neighbors = self.neighbors.len();
        let new_num_neighbors = old_num_neighbors + 1;
        let mut rng = rand::rng();
        let scale = (1.0 / (self.d_head as f32)).sqrt();

        let mut new_weights = vec![0.0f32; self.d_head * new_num_neighbors];

        for d in 0..self.d_head {
            for k in 0..old_num_neighbors {
                new_weights[d * new_num_neighbors + k] = self.weights[d * old_num_neighbors + k];
            }
            new_weights[d * new_num_neighbors + old_num_neighbors] =
                rng.random_range(-scale..scale);
        }

        self.neighbors.push(neighbor_id);
        self.weights = new_weights;
    }

    /// Prune dead/inactive routing links whose max weight norm falls below threshold
    pub fn prune_dead_links(&mut self, pruning_threshold: f32) {
        let num_neighbors = self.neighbors.len();
        if num_neighbors <= 1 {
            return;
        }

        let mut keep_indices = Vec::new();
        let mut max_norm = -1.0f32;
        let mut best_k = 0;

        for k in 0..num_neighbors {
            let mut norm_sq = 0.0f32;
            for d in 0..self.d_head {
                let w = self.weights[d * num_neighbors + k];
                norm_sq += w * w;
            }
            let norm = (norm_sq / (self.d_head as f32)).sqrt();
            if norm > max_norm {
                max_norm = norm;
                best_k = k;
            }
            if norm >= pruning_threshold {
                keep_indices.push(k);
            }
        }

        if keep_indices.is_empty() {
            keep_indices.push(best_k);
        }

        if keep_indices.len() == num_neighbors {
            return;
        }

        let new_num_neighbors = keep_indices.len();
        let mut new_neighbors = Vec::with_capacity(new_num_neighbors);
        let mut new_weights = vec![0.0f32; self.d_head * new_num_neighbors];

        for (new_k, &old_k) in keep_indices.iter().enumerate() {
            new_neighbors.push(self.neighbors[old_k]);
            for d in 0..self.d_head {
                new_weights[d * new_num_neighbors + new_k] =
                    self.weights[d * num_neighbors + old_k];
            }
        }

        self.neighbors = new_neighbors;
        self.weights = new_weights;
    }

    /// Predict next hop neighbor node index using Q-Routing dot product + Softmax (0 heap allocations)
    pub fn select_next_hop(&self, particle: &Particle, temperature: f32) -> usize {
        let num_neighbors = self.neighbors.len();
        if num_neighbors == 0 {
            return 0;
        }

        let n_clamped = num_neighbors.min(64);
        let mut logits = [0.0f32; 64];
        let mut exps = [0.0f32; 64];

        let temp_inv = 1.0 / temperature.max(1e-4);
        let mut max_logit = f32::NEG_INFINITY;

        for k in 0..n_clamped {
            let mut dot = 0.0f32;
            for d in 0..self.d_head {
                dot += particle.payload[d] * self.weights[d * num_neighbors + k];
            }
            let l = dot * temp_inv;
            logits[k] = l;
            if l > max_logit {
                max_logit = l;
            }
        }

        let mut sum_exp = 0.0f32;
        for k in 0..n_clamped {
            let e = (logits[k] - max_logit).exp();
            exps[k] = e;
            sum_exp += e;
        }

        let mut rng = rand::rng();
        let p: f32 = rng.random_range(0.0..1.0);
        let mut cum_sum = 0.0f32;
        let inv_sum = 1.0 / (sum_exp + 1e-8);

        for k in 0..n_clamped {
            cum_sum += exps[k] * inv_sum;
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

    /// Count total active P2P links across all routing tables
    pub fn total_links(&self) -> usize {
        self.routing_tables
            .iter()
            .map(|rt| rt.neighbors.len())
            .sum()
    }

    /// Execute synaptic link pruning across all local routing tables, returning (total_before, total_pruned)
    pub fn prune_all_links(&mut self, pruning_threshold: f32) -> (usize, usize) {
        let mut total_before = 0;
        let mut total_pruned = 0;
        for rt in self.routing_tables.iter_mut() {
            let before = rt.neighbors.len();
            rt.prune_dead_links(pruning_threshold);
            let after = rt.neighbors.len();
            total_before += before;
            total_pruned += before.saturating_sub(after);
        }
        (total_before, total_pruned)
    }
}
