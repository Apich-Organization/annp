use annp_core::{OnlineStats, Particle};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// # RoutingTable — Local Q-Routing table for P2P particle forwarding
///
/// Each MicroBlockNode maintains its own RoutingTable. Routing is decentralized:
/// no node knows about other nodes' routing tables; each routes based solely on
/// local content affinity and its own empirical credit observations.
///
/// ## Routing Decision: `dot + Thompson_Sample(credit)`
///
/// Score for each neighbor k:
///   score_k = dot(p, W_k) + credit_mean_k + SE_k * t_sample(df_k)
///
/// **WHY NOT pure `softmax(dot(p, W))`?**
/// - `dot(p, W_k)` measures content affinity: "does this particle's embedding look
///   like what neighbor k typically processes?" It captures STRUCTURE routing.
/// - `credit_mean + SE * t_sample` is Thompson Sampling over PERFORMANCE routing:
///   it reflects whether particles that went to k recently came back with positive ΔR.
/// - Combining both prevents two failure modes:
///   1. Pure content routing ignores whether a neighbor is actually useful (low credit).
///   2. Pure credit routing ignores whether the particle's content is appropriate for k.
///
/// **WHY Thompson Sampling NOT UCB?**
/// - UCB: score = mean + c * SE requires tuning `c` (exploration coefficient).
/// - Thompson Sampling: sample from the posterior; uncertainty drives exploration
///   automatically. No hyperparameter, and exploration self-regulates as credit
///   statistics accumulate.
///
/// ## Weight Matrix
/// `weights` [d_head, num_neighbors]: learned content affinity vectors per neighbor.
/// Initialized from U(-scale, scale) where scale = 1/sqrt(d_head).
/// Updated implicitly via credit feedback through `observe_credit`.
/// (Note: weights are NOT gradient-updated; they remain as random content projections.
///  All learning happens through the `edge_credit` EMA statistics.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTable {
    pub d_head: usize,
    pub neighbors: Vec<usize>,
    /// Content affinity matrix, shape [d_head, num_neighbors], row-major.
    pub weights: Vec<f32>,
    /// Per-edge credit statistics (OnlineStats EMA Welford) for Thompson Sampling.
    /// Populated lazily (empty until first observation).
    #[serde(default)]
    pub edge_credit: Vec<OnlineStats>,
}

impl RoutingTable {
    pub fn new_with_rng<R: rand::Rng + ?Sized>(
        d_head: usize,
        neighbors: Vec<usize>,
        rng: &mut R,
    ) -> Self {
        let num_neighbors = neighbors.len();
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

    pub fn new(d_head: usize, neighbors: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        Self::new_with_rng(d_head, neighbors, &mut rng)
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

    /// Select next hop using content affinity + Thompson Sampling over edge credit.
    ///
    /// Score for neighbor k:
    ///   score_k = dot(p.payload, W_k) + credit_thompson_sample_k
    ///
    /// New edges (count ≤ 1) use `prior_mean` as their credit estimate.
    /// WHY PRIOR_MEAN NOT ZERO?
    /// - Zero initialization is pessimistic: a new edge would always score below
    ///   established edges with positive mean credit, and would rarely be explored.
    /// - `prior_mean` = average of all known edges' means — an optimistic/neutral
    ///   starting point that gives new edges a fair chance of selection on first few
    ///   visits. This is the multi-armed bandit "optimistic initialization" principle.
    pub fn select_next_hop(&self, particle: &Particle) -> usize {
        let num_neighbors = self.neighbors.len();
        if num_neighbors == 0 {
            return 0;
        }

        // --- Student-t Thompson Sampling for Routing ---
        // Score = Content_Similarity + Thompson_Sample(Credit_Distribution)

        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;

        // Compute prior mean over edges with enough data (count > 1).
        // Used as credit estimate for unvisited edges instead of 0.
        let valid_stats: Vec<_> = self.edge_credit.iter().filter(|s| s.count > 1.0).collect();
        let prior_mean = if valid_stats.is_empty() {
            0.0
        } else {
            valid_stats.iter().map(|s| s.mean).sum::<f32>() / (valid_stats.len() as f32)
        };

        for k in 0..num_neighbors {
            let mut dot = 0.0f32;
            for d in 0..self.d_head {
                dot += particle.payload[d] * self.weights[d * num_neighbors + k];
            }

            let score = if let Some(stats) = self.edge_credit.get(k) {
                if stats.count <= 1.0 {
                    // New edge: use prior_mean as neutral credit estimate.
                    // Gets a fair first-visit score, not pessimistically zero.
                    dot + prior_mean
                } else {
                    let mean = stats.mean;
                    let se = stats.standard_error();
                    let df = (stats.count - 1.0).max(1.0);
                    let u1: f32 = rand::random::<f32>();
                    let u2: f32 = rand::random::<f32>();
                    let t_sample = annp_core::student_t_sample_approximation(df, u1, u2);
                    // Thompson Sample: draw from t-distribution around empirical mean.
                    // Heavy tails at low df (early learning) enable rare explorations;
                    // converges to Normal as df grows (exploitation dominates).
                    dot + mean + se * t_sample
                }
            } else {
                dot + prior_mean
            };

            if score > best_score {
                best_score = score;
                best = k;
            }
        }
        self.neighbors[best]
    }

    /// Update routing table with observed credit and perform Hebbian update on positive credit.
    ///
    /// ## Hebbian Routing Weight Update Rule:
    ///
    ///   W_k = clamp(W_k + eta_r * credit * p_out_normed, -max_w, max_w)
    ///   eta_r = 0.01 / d_head
    ///   max_w = 3.0 / sqrt(d_head)
    ///
    /// ### Mathematical Rationale:
    /// Without dynamic updating, content routing weights W_k remain fixed random projections.
    /// In high-Gini traffic concentration (hub nodes), fixed W_k routes particles blindly based on
    /// static projection rather than learned transformation quality. Updating W_k along positive
    /// credit directions (credit > 0) turns edge k into a dynamic semantic filter that aligns
    /// with the particle manifold that node i successfully processes, preventing hub overload.
    ///
    /// ### Learning Rate Scaling (eta_r = 0.01 / d_head):
    /// Scale 1/d_head matches the dimension normalization of particle payloads (RMS ≈ 1).
    /// Coefficient 0.01 provides smooth, gradual alignment without overwhelming Thompson credit.
    pub fn observe_credit(&mut self, selected_neighbor: usize, credit: f32, payload: &[f32]) {
        self.ensure_edge_credit();
        if let Some(index) = self
            .neighbors
            .iter()
            .position(|&id| id == selected_neighbor)
        {
            self.edge_credit[index].observe(credit);

            // Path A: Hebbian routing weight update on positive credit
            if credit > 0.0 && payload.len() == self.d_head {
                let sq_sum: f32 = payload.iter().map(|&x| x * x).sum();
                let inv_rms = 1.0 / (sq_sum / (self.d_head as f32) + annp_core::RMS_EPSILON).sqrt();
                let eta_r = 0.01 / (self.d_head as f32);
                let num_neighbors = self.neighbors.len();
                let max_w = (1.0 / (self.d_head as f32)).sqrt() * 3.0;

                for (d, &p_val) in payload.iter().enumerate().take(self.d_head) {
                    let p_normed = p_val * inv_rms;
                    let idx = d * num_neighbors + index;
                    let updated = self.weights[idx] + eta_r * credit * p_normed;
                    self.weights[idx] = updated.clamp(-max_w, max_w);
                }
            }
        }
    }

    /// Prune statistically dominated routing links using 2σ confidence intervals.
    ///
    /// A link is dominated if its 95% CI upper bound is below the best link's lower bound:
    ///   upper_k = mean_k + 2*SE_k  <  best_lower = max_j(mean_j - 2*SE_j)
    ///
    /// WHY 2σ CONFIDENCE INTERVALS NOT DIRECT MEAN COMPARISON?
    /// - Direct comparison (mean_k < best_mean) prunes links that appear worse just
    ///   due to noise, even if their true performance is comparable.
    /// - 2σ CI requires statistical significance: we only prune when 95% confident
    ///   the link is truly inferior, not just transiently below average.
    /// - Links with few observations (high SE) get wide CIs → never pruned early.
    ///   This gives new and infrequently visited links sufficient exploration time.
    ///
    /// We always keep at least 1 link (even if all have negative credit), because
    /// a node without routing options cannot forward particles at all.
    pub fn prune_dominated_links(&mut self) -> usize {
        self.ensure_edge_credit();
        if self.neighbors.len() <= 1 {
            return 0;
        }
        let n = self.neighbors.len();
        // Best lower CI bound: the strongest evidence a link is good.
        let best_lower = self
            .edge_credit
            .iter()
            .map(|stats| {
                if stats.count <= 1.0 {
                    f32::NEG_INFINITY // Insufficient data: don't use as reference
                } else {
                    stats.mean - stats.standard_error() * 2.0
                }
            })
            .fold(f32::NEG_INFINITY, f32::max);
        let keep: Vec<usize> = self
            .edge_credit
            .iter()
            .enumerate()
            .filter_map(|(k, stats)| {
                let upper = if stats.count <= 1.0 {
                    f32::INFINITY // Insufficient data: always keep (unexplored)
                } else {
                    stats.mean + stats.standard_error() * 2.0
                };
                (upper >= best_lower).then_some(k) // Keep if not statistically dominated
            })
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
///
/// The grid uses a structured hypercube-like connectivity pattern to balance:
/// - **Short path length**: O(log N) hops between any two nodes.
/// - **Topological diversity**: Multiple structurally distinct paths between any pair.
///
/// ## Why NOT simple ring (+1, +2, ...) connectivity?
/// Ring topology creates a 1D linear flow: particles travel along a single "highway".
/// At N=100 nodes, the diameter is 50 hops (half the ring), even with k=8 neighbors.
///
/// The structured step pattern (base_offset = sqrt(N), varying multipliers) creates
/// connections to nodes at distances sqrt(N), 2*sqrt(N), 3*sqrt(N), ... (mod N),
/// analogous to a hypercube's dimension-based connectivity. This reduces diameter to
/// O(sqrt(N)) steps with the same number of neighbors, enabling richer particle flow.
pub struct TopologyGrid {
    pub num_nodes: usize,
    pub routing_tables: Vec<RoutingTable>,
}

impl TopologyGrid {
    pub fn new_with_rng<R: rand::Rng + ?Sized>(
        num_nodes: usize,
        d_head: usize,
        neighbors_per_node: usize,
        rng: &mut R,
    ) -> Self {
        let mut routing_tables = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            let mut neighbors = Vec::with_capacity(neighbors_per_node);

            // Structured mesh generation: connect node i to nodes at positions
            //   (i + base_offset * k) mod num_nodes  for k = 1, 2, 3, ...
            // where base_offset = ceil(sqrt(num_nodes)).
            //
            // WHY NOT simple ring (+1, +2)?
            // Ring topology creates a 1D linear flow. Particles must traverse O(N/2) hops
            // to reach far nodes even with k neighbors. The sqrt(N)-based step creates
            // a hypercube-like structure: diameter reduces to O(sqrt(N)) and the graph
            // becomes well-connected, enabling rich multi-path particle flow.
            // This is analogous to how a 2D grid has sqrt(N) diameter vs. O(N) for a ring.
            let base_offset = (num_nodes as f32).sqrt().ceil() as usize;
            let mut step_mult = 1;
            while neighbors.len() < neighbors_per_node && step_mult < num_nodes {
                let step = base_offset.wrapping_mul(step_mult).max(1);
                let neighbor_id = (i + step) % num_nodes;
                if neighbor_id != i && !neighbors.contains(&neighbor_id) {
                    neighbors.push(neighbor_id);
                }
                step_mult += 1;
            }

            // Fallback to sequential scanning if base_offset math caused cycles
            let mut j = 1;
            while neighbors.len() < neighbors_per_node && j < num_nodes {
                let neighbor_id = (i + j) % num_nodes;
                if neighbor_id != i && !neighbors.contains(&neighbor_id) {
                    neighbors.push(neighbor_id);
                }
                j += 1;
            }
            routing_tables.push(RoutingTable::new_with_rng(d_head, neighbors, rng));
        }

        Self {
            num_nodes,
            routing_tables,
        }
    }

    pub fn new(num_nodes: usize, d_head: usize, neighbors_per_node: usize) -> Self {
        let mut rng = rand::rng();
        Self::new_with_rng(num_nodes, d_head, neighbors_per_node, &mut rng)
    }

    /// Count total active P2P links across all routing tables
    pub fn total_links(&self) -> usize {
        self.routing_tables
            .iter()
            .map(|rt| rt.neighbors.len())
            .sum()
    }
}
