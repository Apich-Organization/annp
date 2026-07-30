use annp_core::{MicroBlockConfig, NormStrategy};
use annp_model::ANNPModel;
use annp_trainer::{Stage0WaveTrainer, Stage1HardeningTrainer};
use candle_core::{Device, Tensor};

fn main() -> candle_core::Result<()> {
    println!("=== Initializing Asynchronous Neural Network Protocol (ANNP) PoC ===");

    let num_shards = 4;
    let config = MicroBlockConfig {
        num_shards: 4,
        mesh_rows: 4,
        mesh_cols: 4,
        d_head: 64,
        ffn_expansion: 8,
        initial_energy: 1.0,
        max_hop: 100,
        min_hop: 5,
        norm_strategy: NormStrategy::MicroRMSNorm,
        subnode_max: 8,
    };

    let device = Device::Cpu;
    let mut model = ANNPModel::new(config.num_nodes(), num_shards, config, device.clone());

    // Generate simulated Token Embeddings for seq_len = 8, d_model = 4 * 64 = 256
    let seq_len = 8;
    let d_model = num_shards * 64;
    let input_embeddings = Tensor::randn(0.0f32, 1.0f32, (seq_len, d_model), &device)?;

    println!(
        "Input Sequence Tensor Shape: {:?}",
        input_embeddings.shape()
    );

    // Forward Pass Demonstration
    let output_embeddings = model.forward(&input_embeddings)?;
    println!(
        "Output Sequence Tensor Shape: {:?}",
        output_embeddings.shape()
    );

    println!("\n=== Running Streamlined 2-Stage Evolutionary Trainer Demonstration ===");

    // Stage 0: Global Wave Exploration
    let mut stage0 = Stage0WaveTrainer::new(1e-3);
    let loss0 = stage0.train_step(&mut model, &input_embeddings)?;
    println!("[Stage 0: Global Wave Exploration] Loss: {:.6}", loss0);

    // Stage 1: Plastic Hardening & Precision Fine-Tuning
    let stage1 = Stage1HardeningTrainer::new(1e-3, 0.001, 1.5);
    stage1.apply_plastic_hardening(&mut model);
    println!(
        "[Stage 1: Plastic Hardening] Applied to all {} Micro-Block nodes.",
        model.num_nodes
    );

    println!("\nANNP Streamlined Pipeline successfully executed with industrial standards!");
    use std::io::Write;
    std::io::stdout().flush().unwrap();
    Ok(())
}
