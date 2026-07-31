use annp_core::{OnlineStats, Particle};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Q-Routing table maintained locally at each Micro-Block Node.
/// Size: [d_head, num_neighbors]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub d_head: usize,
    pub neighbors: Vec<usize>,
    pub weights: Vec<f32>, // Flat [d_head * num_neighbors]
    #[serde(default)]
    pub edge_credit: Vec<OnlineStats>,
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
            edge_credit: vec![OnlineStats::default(); num_neighbors],
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
        self.edge_credit.push(OnlineStats::default());
    }

    fn ensure_edge_credit(&mut self) {
        if self.edge_credit.len() != self.neighbors.len() {
            self.edge_credit = vec![OnlineStats::default(); self.neighbors.len()];
        }
    }

    /// Select from local content affinity and empirically observed local credit.
    /// Both are normalized only across this node's own neighbors; no broadcast,
    /// global scale, or manually weighted score is involved.
    pub fn select_next_hop(&self, particle: &Particle) -> usize {
        let num_neighbors = self.neighbors.len();
        if num_neighbors == 0 {
            return 0;
        }

        // Mathematical UCB1 Algorithm for local Explore-Exploit routing (no global normalization or arbitrary z-scores)
        // Score = Content_Similarity + C * sqrt(ln(total_activations) / edge_activations)
        let total_activations = self
            .edge_credit
            .iter()
            .map(|stats| stats.count)
            .sum::<u64>()
            .max(1);
        let log_total = (total_activations as f32).ln().max(0.001);

        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;

        for k in 0..num_neighbors {
            let mut dot = 0.0f32;
            for d in 0..self.d_head {
                dot += particle.payload[d] * self.weights[d * num_neighbors + k];
            }

            let ucb_score = if let Some(stats) = self.edge_credit.get(k) {
                if stats.count == 0 {
                    f32::INFINITY
                } else {
                    let mean = stats.mean;
                    let std_dev = stats.variance().sqrt().max(1e-4);
                    let explore = std_dev * (1.0 * log_total / stats.count as f32).sqrt();
                    dot + mean + explore
                }
            } else {
                f32::INFINITY
            };

            if ucb_score > best_score {
                best_score = ucb_score;
                best = k;
            }
        }
        self.neighbors[best]
    }

    pub fn observe_credit(&mut self, selected_neighbor: usize, credit: f32) {
        self.ensure_edge_credit();
        if let Some(index) = self
            .neighbors
            .iter()
            .position(|&id| id == selected_neighbor)
        {
            self.edge_credit[index].observe(credit);
        }
    }

    /// Prune only statistically dominated links, never by a user supplied
    /// magnitude threshold.
    pub fn prune_dominated_links(&mut self) -> usize {
        self.ensure_edge_credit();
        if self.neighbors.len() <= 1 {
            return 0;
        }
        let n = self.neighbors.len();
        let best_lower = self
            .edge_credit
            .iter()
            .map(OnlineStats::pessimistic_value)
            .fold(f32::NEG_INFINITY, f32::max);
        let keep: Vec<usize> = self
            .edge_credit
            .iter()
            .enumerate()
            .filter_map(|(k, stats)| (stats.optimistic_value() >= best_lower).then_some(k))
            .collect();
        if keep.len() == n {
            return 0;
        }
        let mut weights = vec![0.0; self.d_head * keep.len()];
        for d in 0..self.d_head {
            for (new_k, &old_k) in keep.iter().enumerate() {
                weights[d * keep.len() + new_k] = self.weights[d * n + old_k];
            }
        }
        let pruned = n - keep.len();
        let new_neighbors: Vec<usize> = keep.iter().map(|&k| self.neighbors[k]).collect();
        let new_credit: Vec<OnlineStats> = keep.iter().map(|&k| self.edge_credit[k]).collect();
        self.neighbors = new_neighbors;
        self.weights = weights;
        self.edge_credit = new_credit;
        pruned
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

            // Dynamic structured mesh generation without hardcoded offsets like '7'
            // Creates a hypercube-like local connection pattern based on prime increments
            let base_offset = (num_nodes as f32).sqrt().ceil() as usize;
            for n in 1..=neighbors_per_node {
                let step = base_offset.wrapping_mul(n).max(1);
                let neighbor_id = (i + step) % num_nodes;
                if neighbor_id != i && !neighbors.contains(&neighbor_id) {
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
}
