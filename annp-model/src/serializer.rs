use annp_core::Particle;
use candle_core::{Device, Result, Tensor};

/// Egress Serializer / Receiver:
/// Collects settled particles from P2P topology mesh, sorts them by (origin_token_id, shard_id),
/// concatenates particles into full d_model embeddings [seq_len, d_model], and collapses back to tokens.
pub struct EgressSerializer {
    pub d_head: usize,
    pub num_shards: usize,
}

impl EgressSerializer {
    pub fn new(d_head: usize, num_shards: usize) -> Self {
        Self { d_head, num_shards }
    }

    /// Reconstruct sequence tensor [seq_len, d_model] from halted particles
    pub fn reconstruct_sequence(
        &self,
        seq_len: usize,
        halted_particles: &[Particle],
        device: &Device,
    ) -> Result<Tensor> {
        let d_model = self.d_head * self.num_shards;
        let mut full_data = vec![0.0f32; seq_len * d_model];

        for p in halted_particles {
            let t = p.header.origin_token_id as usize;
            let shard = p.header.shard_id as usize;

            if t < seq_len && shard < self.num_shards {
                let token_offset = t * d_model;
                let shard_offset = token_offset + shard * self.d_head;

                for d in 0..self.d_head {
                    if d < p.payload.len() {
                        full_data[shard_offset + d] = p.payload[d];
                    }
                }
            }
        }

        Tensor::from_vec(full_data, (seq_len, d_model), device)
    }
}
