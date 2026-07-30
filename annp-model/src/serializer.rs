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
    /// Pre-normalization egress activations.  The network's public representation
    /// is RMS bounded, so this cache is required to apply the corresponding
    /// Jacobian during training.
    pub last_projected_data: Vec<f32>,
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
            last_projected_data: Vec::new(),
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
                    if j / self.d_head != i / self.d_head {
                        continue;
                    }
                    sum += full_data[t * d_model + j] * self.w_egress[j * d_model + i];
                }
                proj_data[t * d_model + i] = sum;
            }
        }

        // A particle may legitimately traverse many micro-blocks before its
        // energy expires.  MicroRMSNorm bounds each *residual update*, not the
        // accumulated particle state.  The prediction space must therefore be
        // bounded explicitly: dataset embeddings are unit-RMS and the initial
        // MSE of two unrelated unit-RMS vectors is approximately 2.
        self.last_projected_data = proj_data.clone();
        for row in proj_data.chunks_exact_mut(d_model) {
            let rms = (row.iter().map(|x| x * x).sum::<f32>() / d_model as f32 + 1e-8).sqrt();
            for value in row {
                *value /= rms;
            }
        }

        Tensor::from_vec(proj_data, (seq_len, d_model), device)
    }

    /// Backpropagate a gradient through the per-token RMS projection performed
    /// by `reconstruct_sequence`. `output_gradient` has the same shape as the
    /// serialized output and is returned as a gradient for the dense projection.
    pub fn backprop_output_rms(&self, output_gradient: &[f32]) -> Vec<f32> {
        let d_model = self.d_head * self.num_shards;
        if d_model == 0
            || output_gradient.len() != self.last_projected_data.len()
            || output_gradient.len() % d_model != 0
        {
            return Vec::new();
        }

        let mut projected_gradient = vec![0.0; output_gradient.len()];
        for ((z_row, grad_row), out_row) in self
            .last_projected_data
            .chunks_exact(d_model)
            .zip(output_gradient.chunks_exact(d_model))
            .zip(projected_gradient.chunks_exact_mut(d_model))
        {
            let rms = (z_row.iter().map(|x| x * x).sum::<f32>() / d_model as f32 + 1e-8).sqrt();
            let dot_over_dim = z_row
                .iter()
                .zip(grad_row)
                .map(|(z, grad)| z * grad)
                .sum::<f32>()
                / d_model as f32;
            for ((out, &grad), &z) in out_row.iter_mut().zip(grad_row).zip(z_row) {
                *out = (grad - z * dot_over_dim / (rms * rms)) / rms;
            }
        }
        projected_gradient
    }

    /// Propagate a dense-projection gradient to the reconstructed particle
    /// features: dL/dX = dL/dZ * W_egress^T.
    pub fn input_gradient(&self, projected_gradient: &[f32]) -> Vec<f32> {
        let d_model = self.d_head * self.num_shards;
        if d_model == 0 || projected_gradient.len() % d_model != 0 {
            return Vec::new();
        }

        let mut input_gradient = vec![0.0; projected_gradient.len()];
        for (grad_row, input_row) in projected_gradient
            .chunks_exact(d_model)
            .zip(input_gradient.chunks_exact_mut(d_model))
        {
            for j in 0..d_model {
                let mut sum = 0.0;
                for i in 0..d_model {
                    sum += grad_row[i] * self.w_egress[j * d_model + i];
                }
                input_row[j] = sum;
            }
        }
        input_gradient
    }

    /// Update W_egress matrix using exact X^T * diff matrix product
    pub fn update_weights(&mut self, diff_matrix: &[f32], lr: f32) {
        let d_model = self.d_head * self.num_shards;
        if diff_matrix.len() != self.last_full_data.len() || self.last_full_data.is_empty() {
            return;
        }

        let seq_len = self.last_full_data.len() / d_model;
        for j in 0..d_model {
            for i in 0..d_model {
                if j / self.d_head != i / self.d_head {
                    continue;
                }
                let idx = j * d_model + i;
                let mut grad = 0.0f32;
                for t in 0..seq_len {
                    let x_val = self.last_full_data[t * d_model + j];
                    let diff_val = diff_matrix[t * d_model + i];
                    grad += x_val * diff_val;
                }
                grad /= (seq_len as f32) * (d_model as f32);

                // `grad` is already the mean MSE gradient. Applying it
                // directly keeps the configured learning rate meaningful;
                // momentum, weight decay and a fixed clamp previously reduced
                // an initial update to noise-level magnitude.
                self.w_egress[idx] -= lr * grad;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egress_serializer_reconstruct_and_update() -> Result<()> {
        let d_head = 64;
        let num_shards = 4;
        let d_model = d_head * num_shards;
        let seq_len = 2;

        let mut serializer = EgressSerializer::new(d_head, num_shards);
        let device = Device::Cpu;

        let particles = vec![
            Particle::new(annp_core::ParticleHeader::new(0, 0, 1.0), vec![1.0f32; 64]),
            Particle::new(annp_core::ParticleHeader::new(0, 1, 1.0), vec![2.0f32; 64]),
            Particle::new(annp_core::ParticleHeader::new(0, 2, 1.0), vec![3.0f32; 64]),
            Particle::new(annp_core::ParticleHeader::new(0, 3, 1.0), vec![4.0f32; 64]),
        ];

        let out_tensor = serializer.reconstruct_sequence(seq_len, &particles, &device)?;
        assert_eq!(out_tensor.dims2()?, (seq_len, d_model));

        let diff_matrix = vec![0.1f32; seq_len * d_model];
        serializer.update_weights(&diff_matrix, 0.01);

        Ok(())
    }
}
