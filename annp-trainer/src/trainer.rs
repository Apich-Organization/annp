use annp_model::ANNPModel;
use candle_core::{Result, Tensor};

/// Unified Stage: Global Wave Pre-training & Plasticity Hardening.
/// Shard-specific exact residual backpropagation with learning rate decay.
pub struct Trainer {
    pub base_lr: f32,
}

impl Trainer {
    pub fn new(base_lr: f32) -> Self {
        Self { base_lr }
    }

    pub fn train_step(&mut self, model: &mut ANNPModel, input_embeddings: &Tensor) -> Result<f32> {
        self.train_step_with_epoch(model, input_embeddings, 0)
    }

    pub fn train_step_with_epoch(
        &mut self,
        model: &mut ANNPModel,
        input_embeddings: &Tensor,
        _epoch_idx: usize,
    ) -> Result<f32> {
        let (full_seq_len, _d_model) = input_embeddings.dims2()?;

        // Reset node KV Caches before each sequence to prevent cross-sequence pollution
        model.reset_state();

        let mut final_seq_loss = 0.0;

        // Feed tokens sequentially to allow temporal difference learning to emerge from the particle flow
        for i in 0..full_seq_len {
            let single_token = input_embeddings.narrow(0, i, 1)?;
            let (_, step_loss) = model.forward(&single_token, i, Some(self.base_lr))?;
            final_seq_loss = step_loss; // model.forward returns the cumulative average loss across all nodes
        }

        Ok(final_seq_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_core::MicroBlockConfig;
    use candle_core::Device;

    fn create_test_config() -> MicroBlockConfig {
        MicroBlockConfig {
            num_shards: 4,
            mesh_rows: 2,
            mesh_cols: 2,
            d_head: 64,
            ffn_expansion: 8,
            initial_energy: 1.0,
            max_hop: 20,
            min_hop: 2,
            subnode_max: 8,
            weight_decay: 1e-4,
            ingress_ratio: 0.1,
            k_neighbors: 4,
        }
    }

    #[test]
    fn test_stage0_train_step() -> Result<()> {
        let config = create_test_config();
        let device = Device::Cpu;
        let mut model = ANNPModel::new_with_cuda(4, 4, config, device.clone(), false);

        let d_model = 4 * 64;
        let tensor_data = vec![0.5f32; 2 * d_model];
        let input_embeddings = Tensor::from_vec(tensor_data, (2, d_model), &device)?;

        let mut trainer = Trainer::new(2.0);
        let loss = trainer.train_step_with_epoch(&mut model, &input_embeddings, 0)?;
        println!("DEBUG_TEST_LOSS = {}", loss);

        assert!(loss >= 0.0);
        assert!(loss.is_finite());

        Ok(())
    }
}
