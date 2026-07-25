use annp_model::ANNPModel;
use candle_core::{Result, Tensor};

/// Stage 2: Energy Settling & Pondering Cost Training.
/// Optimizes Test-time compute depth with Ponder Loss penalty: \lambda_ponder * \sum Hop_i.
pub struct Stage2PonderTrainer {
    pub lambda_ponder: f32,
}

impl Stage2PonderTrainer {
    pub fn new(lambda_ponder: f32) -> Self {
        Self { lambda_ponder }
    }

    pub fn train_step(&mut self, model: &mut ANNPModel, input_embeddings: &Tensor) -> Result<(f32, f32)> {
        let output = model.forward(input_embeddings)?;
        let task_loss = output.sqr()?.mean_all()?.to_scalar::<f32>()?;

        // Total average hop count across nodes
        let total_activations: u64 = model.nodes.iter().map(|n| n.activation_count).sum();
        let avg_hops = total_activations as f32 / (model.num_nodes as f32 + 1e-5);
        let ponder_loss = self.lambda_ponder * avg_hops;

        let total_loss = task_loss + ponder_loss;
        Ok((total_loss, avg_hops))
    }
}
