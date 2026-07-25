use serde::{Deserialize, Serialize};

/// Lightweight metadata header attached to each particle shard during P2P routing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ParticleHeader {
    /// Token identifier from original sequence (0..L-1)
    pub origin_token_id: u32,
    /// Particle shard index (0..N-1)
    pub shard_id: u16,
    /// Current remaining energy (E_init -> 0.0)
    pub energy: f32,
    /// Accumulated hop counter
    pub hop_count: u16,
    /// Settling / Halting flag (true if particle triggered spontaneous or forced halting)
    pub halted: bool,
}

impl ParticleHeader {
    pub fn new(origin_token_id: u32, shard_id: u16, initial_energy: f32) -> Self {
        Self {
            origin_token_id,
            shard_id,
            energy: initial_energy,
            hop_count: 0,
            halted: false,
        }
    }

    /// Step hop counter and deduct energy delta = 1 / max_hop
    pub fn step_hop(&mut self, max_hop: u16) {
        self.hop_count += 1;
        let delta_e = 1.0 / (max_hop as f32);
        self.energy = (self.energy - delta_e).max(0.0);
        if self.energy <= 0.0 || self.hop_count >= max_hop {
            self.halted = true;
        }
    }
}

/// Token particle shard containing metadata header and d_head floating-point payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub header: ParticleHeader,
    pub payload: Vec<f32>,
}

impl Particle {
    pub fn new(header: ParticleHeader, payload: Vec<f32>) -> Self {
        Self { header, payload }
    }

    pub fn d_head(&self) -> usize {
        self.payload.len()
    }
}
