use annp_model::ANNPModel;
use candle_core::{Result, Tensor};

/// Stage 0: Embryonic Stage - Global Wave Pre-training.
/// Shard-specific exact residual backpropagation with learning rate decay.
pub struct Stage0WaveTrainer {
    pub base_lr: f32,
}

impl Stage0WaveTrainer {
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
        epoch_idx: usize,
    ) -> Result<f32> {
        let (full_seq_len, d_model) = input_embeddings.dims2()?;
        let d_head = model.config.d_head;
        let num_shards = d_model / d_head;

        // Autoregressive Next-Token Slicing: Inputs = X[0..S-1], Targets = X[1..S]
        let (inputs, targets) = if full_seq_len >= 2 {
            let inp = input_embeddings.narrow(0, 0, full_seq_len - 1)?;
            let tgt = input_embeddings.narrow(0, 1, full_seq_len - 1)?;
            (inp, tgt)
        } else {
            (input_embeddings.clone(), input_embeddings.clone())
        };

        let output = model.forward(&inputs)?;
        let (seq_len, _) = inputs.dims2()?;

        // Learning rate decay: lr = base_lr * (0.85)^epoch_idx
        let lr_decay = self.base_lr * 0.85f32.powi(epoch_idx as i32);

        // Autoregressive Next-Token Prediction MSE Loss (output vs targets)
        let diff = (output.clone() - &targets)?;
        let mse_loss = (diff.sqr()?.sum_all()?.to_scalar::<f32>()?) / (targets.elem_count() as f32);

        let diff_vec = diff.flatten_all()?.to_vec1::<f32>()?;

        // `forward` returns RMS-bounded egress vectors. We apply the gradient directly 
        // to the shard errors since the output nodes now emit directly without projection.
        let input_grad = diff_vec;

        // Broadcast the exact residual error to all active nodes.
        // Each node will correlate this global error with its local sequence history.
        for node in model.nodes.iter_mut() {
            node.update_weights_from_broadcast_error(&input_grad, seq_len, d_model, lr_decay);
        }

        Ok(mse_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_core::{MicroBlockConfig, NormStrategy};
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
            norm_strategy: NormStrategy::MicroRMSNorm,
            subnode_max: 8,
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

        let mut trainer = Stage0WaveTrainer::new(0.02);
        let loss = trainer.train_step_with_epoch(&mut model, &input_embeddings, 0)?;
        println!("DEBUG_TEST_LOSS = {}", loss);

        assert!(loss >= 0.0);
        assert!(loss.is_finite());

        Ok(())
    }
}
