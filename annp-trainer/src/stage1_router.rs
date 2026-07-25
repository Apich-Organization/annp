use annp_model::ANNPModel;
use candle_core::{Result, Tensor};

/// Stage 1: Synaptic Pruning & Router Auto-organization.
/// Transition from soft routing to Gumbel-Softmax discrete routing + Mutual Information Maximization Loss.
pub struct Stage1RouterTrainer {
    pub tau: f32,
    pub mi_weight: f32,
}

impl Stage1RouterTrainer {
    pub fn new(tau: f32, mi_weight: f32) -> Self {
        Self { tau, mi_weight }
    }

    pub fn train_step(&mut self, model: &mut ANNPModel, input_embeddings: &Tensor) -> Result<f32> {
        model.config.temperature = self.tau;
        let output = model.forward(input_embeddings)?;

        // Compute Task Loss + Mutual Information Loss proxy
        let task_loss = output.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let mi_loss = self.mi_weight * (self.tau.log2());
        let total_loss = task_loss - mi_loss;

        Ok(total_loss)
    }
}
