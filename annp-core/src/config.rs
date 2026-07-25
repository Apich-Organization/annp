use serde::{Deserialize, Serialize};

/// Configuration for ANNP Micro-Block Nodes and Topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroBlockConfig {
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
    /// Threshold for feature delta change norm ||p_out - p_in||_2
    pub epsilon_p: f32,
    /// Threshold for local attention distribution entropy H
    pub epsilon_h: f32,
    /// Softmax temperature for Q-routing or Gumbel-Softmax
    pub temperature: f32,
    /// Micro-Norm option selection
    pub norm_strategy: NormStrategy,
    /// RMSNorm learning alpha scaling factor initial value
    pub alpha_init: f32,
    /// Base sphere radius for Sphere Normalization
    pub sphere_radius: f32,
}

impl Default for MicroBlockConfig {
    fn default() -> Self {
        Self {
            d_head: 64,
            ffn_expansion: 8,
            initial_energy: 1.0,
            max_hop: 200,
            min_hop: 10,
            epsilon_p: 1e-4,
            epsilon_h: 0.05,
            temperature: 1.0,
            norm_strategy: NormStrategy::MicroRMSNorm,
            alpha_init: 0.01,
            sphere_radius: 1.0,
        }
    }
}

/// Flexible Micro-Normalization Strategies defined in Section 2.2 of ANNP design.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormStrategy {
    /// Option A: Micro-RMSNorm + learnable alpha-scaling decay
    MicroRMSNorm,
    /// Option B: Sphere Normalization (projection onto d_head unit sphere)
    SphereNormalization,
}
