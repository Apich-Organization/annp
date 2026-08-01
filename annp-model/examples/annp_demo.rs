use annp_core::MicroBlockConfig;
use annp_model::ANNPModel;
use annp_trainer::Trainer;
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
        subnode_max: 8,
        weight_decay: 1e-4,
        ingress_ratio: 0.1,
        k_neighbors: 4,
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

    let (output_embeddings, _) = model.forward(&input_embeddings, 0, None)?;
    println!(
        "Output Sequence Tensor Shape: {:?}",
        output_embeddings.shape()
    );

    println!("\n=== Running Streamlined Trainer Demonstration ===");

    // Training
    let mut trainer = Trainer::new(1e-3);
    let loss0 = trainer.train_step(&mut model, &input_embeddings)?;
    println!("[Training Step 1] Loss: {:.6}", loss0);

    let loss1 = trainer.train_step(&mut model, &input_embeddings)?;
    println!("[Training Step 2] Loss: {:.6}", loss1);

    println!("\nANNP Streamlined Pipeline successfully executed with industrial standards!");
    use std::io::Write;
    std::io::stdout().flush().unwrap();
    Ok(())
}
