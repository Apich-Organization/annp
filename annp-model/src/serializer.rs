use annp_core::Particle;
use candle_core::{Device, Result, Tensor};
use rand::Rng;

/// Egress Serializer / Receiver:
/// Collects settled particles from P2P topology mesh, sorts them by (origin_token_id, shard_id),
/// concatenates particles into full d_model embeddings [seq_len, d_model], and projects via learnable W_egress matrix.
pub struct EgressSerializer {
    pub d_head: usize,
    pub num_shards: usize,
    pub w_egress: Vec<f32>,       // Flat [d_model * d_model]
    pub v_egress: Vec<f32>,       // Momentum velocity buffer
    pub last_full_data: Vec<f32>, // Cached reconstructed input features for exact matrix gradient
}

impl EgressSerializer {
    pub fn new(d_head: usize, num_shards: usize) -> Self {
        let d_model = d_head * num_shards;
        let mut rng = rand::rng();
        let scale = (2.0 / d_model as f64).sqrt() as f32;

        let mut w_egress = vec![0.0f32; d_model * d_model];
        for i in 0..d_model {
            for j in 0..d_model {
                if i == j {
                    w_egress[i * d_model + j] = 1.0; // Identity initialization
                } else {
                    w_egress[i * d_model + j] = rng.random_range(-scale * 0.05..scale * 0.05);
                }
            }
        }
        let v_egress = vec![0.0f32; d_model * d_model];
        let last_full_data = Vec::new();

        Self {
            d_head,
            num_shards,
            w_egress,
            v_egress,
            last_full_data,
        }
    }

    /// Reconstruct sequence tensor [seq_len, d_model] from halted particles
    pub fn reconstruct_sequence(
        &mut self,
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

        self.last_full_data = full_data.clone();

        // Project full_data through learnable W_egress matrix
        let mut proj_data = vec![0.0f32; seq_len * d_model];
        for t in 0..seq_len {
            for i in 0..d_model {
                let mut sum = 0.0f32;
                for j in 0..d_model {
                    sum += full_data[t * d_model + j] * self.w_egress[j * d_model + i];
                }
                proj_data[t * d_model + i] = sum;
            }
        }

        Tensor::from_vec(proj_data, (seq_len, d_model), device)
    }

    /// Update W_egress matrix using exact X^T * diff matrix product
    pub fn update_weights(&mut self, diff_matrix: &[f32], lr: f32) {
        let d_model = self.d_head * self.num_shards;
        if diff_matrix.len() < d_model || self.last_full_data.is_empty() {
            return;
        }

        let seq_len = self.last_full_data.len() / d_model;
        let beta = 0.9f32;
        let weight_decay = 0.9999f32;

        for j in 0..d_model {
            for i in 0..d_model {
                let idx = j * d_model + i;
                let mut grad = 0.0f32;
                for t in 0..seq_len {
                    let x_val = self.last_full_data[t * d_model + j];
                    let diff_val = diff_matrix[t * d_model + i];
                    grad += x_val * diff_val;
                }
                grad /= seq_len as f32;

                self.v_egress[idx] = beta * self.v_egress[idx] + (1.0 - beta) * grad;
                self.w_egress[idx] =
                    self.w_egress[idx] * weight_decay - lr * self.v_egress[idx].clamp(-0.1, 0.1);
            }
        }
    }
}
