use annp_core::{MicroBlockConfig, Particle, ParticleHeader};
use candle_core::{Result, Tensor};

/// Ingress Scattering Pipeline:
/// Takes Token Embedding matrix [L, d_model] where d_model = N * d_head
/// Physical Scattering splits embedding into N shards of size [L, d_head]
pub struct TokenScattering {
    pub num_shards: usize,
    pub d_head: usize,
    pub ingress_node_indices: Vec<usize>,
}

impl TokenScattering {
    pub fn new(num_shards: usize, d_head: usize, ingress_ratio: f32) -> Self {
        let num_nodes = num_shards; // num_nodes == num_shards in the current architecture
        let num_ingress = ((num_nodes as f32 * ingress_ratio).ceil() as usize).max(1);
        // Uniformly distribute ingress nodes across the full mesh by stepping with stride.
        // Previously took [0..num_ingress] which clusters all ingress in one topological region.
        let step = (num_nodes / num_ingress).max(1);
        let ingress_node_indices: Vec<usize> = (0..num_ingress)
            .map(|i| (i * step).min(num_nodes - 1))
            .collect();

        Self {
            num_shards,
            d_head,
            ingress_node_indices,
        }
    }

    /// Scatter sequence embeddings into particles
    pub fn scatter_embeddings(
        &self,
        embeddings: &Tensor, // Shape [seq_len, d_model]
        config: &MicroBlockConfig,
        offset: usize,
    ) -> Result<Vec<Particle>> {
        let (seq_len, d_model) = embeddings.dims2()?;
        assert_eq!(
            d_model,
            self.num_shards * self.d_head,
            "d_model must equal num_shards * d_head"
        );

        let flat_data = embeddings.flatten_all()?.to_vec1::<f32>()?;
        let mut particles = Vec::with_capacity(seq_len * self.num_shards);

        for t in 0..seq_len {
            let t_offset = t * d_model;
            for shard_i in 0..self.num_shards {
                let start_idx = t_offset + shard_i * self.d_head;
                let end_idx = start_idx + self.d_head;
                let payload = flat_data[start_idx..end_idx].to_vec();

                let header =
                    ParticleHeader::new((t + offset) as u32, shard_i as u16, config.initial_energy);
                particles.push(Particle::new(header, payload));
            }
        }

        Ok(particles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_token_scattering_split() -> Result<()> {
        let num_shards = 4;
        let d_head = 64;
        let d_model = num_shards * d_head;
        let seq_len = 5;

        let scattering = TokenScattering::new(num_shards, d_head, 1.0);
        let config = MicroBlockConfig::default();

        let tensor_data = vec![1.0f32; seq_len * d_model];
        let tensor = Tensor::from_vec(tensor_data, (seq_len, d_model), &Device::Cpu)?;

        let particles = scattering.scatter_embeddings(&tensor, &config, 0)?;

        assert_eq!(particles.len(), seq_len * num_shards);
        for p in &particles {
            assert_eq!(p.d_head(), d_head);
            assert_eq!(p.header.energy, config.initial_energy);
            assert!(!p.header.halted);
        }

        Ok(())
    }
}
