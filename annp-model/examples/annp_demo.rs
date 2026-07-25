use annp_core::{MicroBlockConfig, NormStrategy};
use annp_model::ANNPModel;
use annp_trainer::{
    Stage0WaveTrainer, Stage1RouterTrainer, Stage2PonderTrainer, Stage3ContinualTrainer,
};
use candle_core::{Device, Tensor};

fn main() -> candle_core::Result<()> {
    println!("=== Initializing Asynchronous Neural Network Protocol (ANNP) PoC ===");

    let num_shards = 4;
    let config = MicroBlockConfig {
        mesh_rows: 4,
        mesh_cols: 4,
        d_head: 64,
        ffn_expansion: 8,
        initial_energy: 1.0,
        max_hop: 100,
        min_hop: 5,
        epsilon_p: 1e-4,
        epsilon_h: 0.1,
        temperature: 1.0,
        norm_strategy: NormStrategy::MicroRMSNorm,
        alpha_init: 0.01,
        sphere_radius: 1.0,
        lambda_temporal: 0.001,
        lambda_frequency: 0.01,
        eviction_threshold: 1e-4,
        pruning_threshold: 1e-5,
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

    println!("\n=== Running 4-Stage Evolutionary Trainer Demonstration ===");

    // Stage 0: Global Wave Pre-training
    let mut stage0 = Stage0WaveTrainer::new(1e-3);
    let loss0 = stage0.train_step(&mut model, &input_embeddings)?;
    println!("[Stage 0: Wave Pre-training] Loss: {:.6}", loss0);

    // Stage 1: Router Auto-organization
    let mut stage1 = Stage1RouterTrainer::new(0.8, 0.05);
    let loss1 = stage1.train_step(&mut model, &input_embeddings)?;
    println!("[Stage 1: Router Auto-org] Loss: {:.6}", loss1);

    // Stage 2: Energy Settling & Pondering Cost
    let mut stage2 = Stage2PonderTrainer::new(0.01);
    let (loss2, avg_hops) = stage2.train_step(&mut model, &input_embeddings)?;
    println!(
        "[Stage 2: Energy Settling] Loss: {:.6}, Avg Hops: {:.2}",
        loss2, avg_hops
    );

    // Stage 3: Continual Evolution & Plastic Hardening
    let stage3 = Stage3ContinualTrainer::new(1e-3, 0.001, 1.5);
    stage3.apply_plastic_hardening(&mut model);
    println!(
        "[Stage 3: Plastic Hardening] Applied to all {} Micro-Block nodes.",
        model.num_nodes
    );

    println!("\nANNP PoC Pipeline successfully executed with industrial standards!");
    use std::io::Write;
    std::io::stdout().flush().unwrap();
    Ok(())
}
