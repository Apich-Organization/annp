use serde::{Deserialize, Serialize};

/// Configuration for ANNP Micro-Block Nodes and Topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroBlockConfig {
    /// Number of parallel particle shards scattered from input sequence (default: 4)
    pub num_shards: usize,
    /// Mesh topology dimensions: rows * cols (e.g. 100 * 100 = 10,000 nodes)
    pub mesh_rows: usize,
    pub mesh_cols: usize,
    /// Dimension of single particle vector (e.g., 64 or 128)
    pub d_head: usize,
    /// FFN intermediate layer expansion multiplier (default: 8)
    pub ffn_expansion: usize,
    /// Initial energy given to a newly scattered particle (default: 1.0)
    pub initial_energy: f32,
    /// Hard ceiling on hop count per particle
    pub max_hop: u16,
    /// Minimum hops required before spontaneous halting convergence checks begin
    pub min_hop: u16,
    /// Maximum number of internal subnodes per micro-block container (default: 8)
    pub subnode_max: usize,
    /// L2 Weight decay factor for optimization (default: 1e-4)
    pub weight_decay: f32,
    /// Ratio of nodes to serve as ingress points for scattered particles (default: 0.1)
    pub ingress_ratio: f32,
    /// Number of neighbors for topology routing (default: 4)
    pub k_neighbors: usize,
}

impl Default for MicroBlockConfig {
    fn default() -> Self {
        Self {
            num_shards: 4,
            mesh_rows: 10,
            mesh_cols: 10,
            d_head: 64,
            ffn_expansion: 8,
            initial_energy: 1.0,
            max_hop: 200,
            min_hop: 10,
            subnode_max: 8,
            weight_decay: 1e-4,
            ingress_ratio: 0.1,
            k_neighbors: 4,
        }
    }
}

impl MicroBlockConfig {
    pub fn num_nodes(&self) -> usize {
        self.mesh_rows * self.mesh_cols
    }

    pub fn d_model(&self) -> usize {
        self.num_shards * self.d_head
    }
}
