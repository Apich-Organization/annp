use annp_model::ANNPModel;
use candle_core::{Result, Tensor};

/// Stage 0: Embryonic Stage - Global Wave Pre-training.
/// Particles move in soft wave state, all nodes unfrozen with base learning rate eta_base.
pub struct Stage0WaveTrainer {
    pub base_lr: f32,
}

impl Stage0WaveTrainer {
    pub fn new(base_lr: f32) -> Self {
        Self { base_lr }
    }

    pub fn train_step(&mut self, model: &mut ANNPModel, input_embeddings: &Tensor) -> Result<f32> {
        let output = model.forward(input_embeddings)?;
        let loss = output.sqr()?.sum_all()?.to_scalar::<f32>()?;
        Ok(loss)
    }
}
