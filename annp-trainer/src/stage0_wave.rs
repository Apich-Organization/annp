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
        let output = model.forward(input_embeddings)?;
        let (seq_len, d_model) = input_embeddings.dims2()?;
        let d_head = model.config.d_head;
        let num_shards = d_model / d_head;

        // Learning rate decay: lr = base_lr * (0.85)^epoch_idx
        let lr_decay = self.base_lr * 0.85f32.powi(epoch_idx as i32);

        // Supervised MSE Reconstruction Loss
        let diff = (output.clone() - input_embeddings)?;
        let mse_loss =
            (diff.sqr()?.sum_all()?.to_scalar::<f32>()?) / (input_embeddings.elem_count() as f32);

        let diff_vec = diff.flatten_all()?.to_vec1::<f32>()?;

        // 1. Update Egress Serializer with full diff_vec
        model.serializer.update_weights(&diff_vec, lr_decay);

        // 2. Compute exact Shard-Specific Error Vectors: shard_err[s] of dimension d_head
        let mut shard_errs = vec![vec![0.0f32; d_head]; num_shards];
        for t in 0..seq_len {
            for s in 0..num_shards {
                let base_idx = t * d_model + s * d_head;
                for d in 0..d_head {
                    shard_errs[s][d] += diff_vec[base_idx + d];
                }
            }
        }
        for s in 0..num_shards {
            for d in 0..d_head {
                shard_errs[s][d] /= seq_len as f32;
            }
        }

        // 3. Apply exact shard-specific residual update to active nodes
        for (i, node) in model.nodes.iter_mut().enumerate() {
            let shard_idx = i % num_shards;
            node.update_weights_with_shard_err(&shard_errs[shard_idx], lr_decay);
        }

        Ok(mse_loss)
    }
}
