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
    /// Identifies one independent input excitation without imposing causal
    /// ordering inside that excitation.
    #[serde(default)]
    pub wave_id: u64,
    /// Local contrastive improvement written by the node that last processed
    /// this particle. It is the sole routing-learning signal.
    #[serde(default)]
    pub credit: f32,
    /// `false` means this node had no comparable local context. Absence of a
    /// measurement must never be interpreted as zero or negative evidence.
    #[serde(default)]
    pub credit_valid: bool,
}

impl Particle {
    pub fn new(header: ParticleHeader, payload: Vec<f32>) -> Self {
        Self {
            header,
            payload,
            wave_id: 0,
            credit: 0.0,
            credit_valid: false,
        }
    }

    pub fn d_head(&self) -> usize {
        self.payload.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_header_hop_and_energy_decay() {
        let mut header = ParticleHeader::new(101, 1, 1.0);
        assert_eq!(header.origin_token_id, 101);
        assert_eq!(header.shard_id, 1);
        assert_eq!(header.energy, 1.0);
        assert_eq!(header.hop_count, 0);
        assert!(!header.halted);

        let max_hop = 10;
        for i in 1..=10 {
            header.step_hop(max_hop);
            assert_eq!(header.hop_count, i);
            if i < 10 {
                assert!(!header.halted);
            }
        }

        assert_eq!(header.energy, 0.0);
        assert!(header.halted);
    }

    #[test]
    fn test_particle_payload() {
        let header = ParticleHeader::new(0, 0, 1.0);
        let payload = vec![0.5f32; 64];
        let particle = Particle::new(header, payload);
        assert_eq!(particle.d_head(), 64);
    }
}
