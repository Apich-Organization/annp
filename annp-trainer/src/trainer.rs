use annp_model::ANNPModel;
use candle_core::{Result, Tensor};

/// # Trainer — ANNP Sequential Training Orchestrator
///
/// Feeds token embeddings one-at-a-time to the model. This is the central training
/// loop for ANNP's particle-based learning, distinct from transformer-style batched
/// forward passes.
///
/// ## Why Sequential (Token-by-Token) Not Batched?
///
/// ANNP's TD (Temporal Difference) learning depends on `last_token_id` to track which
/// token preceded the current one. A single batched `forward(seq)` would:
/// 1. Mix particles from all tokens simultaneously, destroying temporal ordering.
/// 2. Prevent per-token `delta_t` computation (1/dt harmonic discount).
/// 3. Make `reset_state()` semantically ambiguous (reset mid-sequence? after?)
///
/// Token-by-token feeding preserves the causal chain required for TD residuals.
///
/// ## Why `reset_state()` Per Sequence, Not Per Epoch?
///
/// `reset_state()` clears ONLY temporal state (`last_p_in`, `last_prediction`,
/// `last_token_id`). It does NOT clear FFN weights or `fast_weight` — those are
/// accumulated knowledge that should persist across the entire training run.
///
/// Epoch = checkpoint interval in ANNP (NOT data repetition). Resetting state per
/// epoch would erase cross-batch TD continuity at arbitrary checkpoint boundaries.
/// Resetting per sequence is the correct boundary: sequence boundaries represent
/// genuine temporal discontinuities where the particle flow context changes.
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

        // Reset temporal state (NOT weights) before each sequence.
        //
        // This clears last_p_in, last_prediction, and last_token_id at all nodes,
        // preventing TD learning from trying to connect the end of one sequence
        // to the beginning of the next (which would be a false temporal association).
        //
        // Crucially, fast_weight and FFN weights are preserved — they represent
        // accumulated learned associations that should NOT be reset between sequences.
        model.reset_state();

        let lr = self.base_lr;
        let mut final_seq_loss = 0.0;

        // Sequential token-by-token feeding — required for TD learning continuity.
        //
        // Each call to model.forward(token_i, i, Some(lr)) allows nodes to:
        // 1. Compute the TD error: actual(t) - predicted(t) using last_prediction.
        // 2. Apply Hebbian updates weighted by 1/dt harmonic discount.
        // 3. Update last_token_id = i for the next iteration's dt computation.
        //
        // The offset `i` becomes `origin_token_id` in scattering, ensuring globally
        // monotone token IDs across the entire training run (preventing false TD
        // associations when resuming — see checkpoint.rs and scattering.rs).
        for i in 0..full_seq_len {
            let single_token = input_embeddings.narrow(0, i, 1)?;
            let (_, step_loss) = model.forward(&single_token, i, Some(lr))?;
            final_seq_loss += step_loss;
        }

        Ok(final_seq_loss / full_seq_len as f32)
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
            health_base: 1.0,
            queue_backpressure: 64,
            step_safety_margin: 20,
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
