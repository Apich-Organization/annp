use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use annp_model::ANNPModel;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use std::time::Instant;

pub fn execute_run(
    config_path: PathBuf,
    checkpoint_path: Option<PathBuf>,
    _input_text: Option<String>,
    benchmark: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading ANNP Configuration from: {:?}", config_path);
    let toml_config = AnnpTomlConfig::load_from_file(config_path)?;
    let core_config = toml_config.to_core_config();

    let device = Device::Cpu;
    let num_shards = 4;
    let mut model = ANNPModel::new(
        core_config.num_nodes(),
        num_shards,
        core_config,
        device.clone(),
    );

    if let Some(ckpt_path) = checkpoint_path {
        println!("Loading Checkpoint: {:?}", ckpt_path);
        let ckpt = ModelCheckpoint::load(ckpt_path)?;
        ckpt.apply_to_model(&mut model);
    }

    let d_model = num_shards * model.config.d_head;
    let seq_len = 16;
    let input_tensor = Tensor::randn(0.0f32, 1.0f32, (seq_len, d_model), &device)?;

    println!("\n=== Executing ANNP Model Inference Pass ===");
    println!(
        "Input Prompt/Vector Tensor Shape: {:?}",
        input_tensor.shape()
    );

    let iterations = if benchmark { 50 } else { 1 };
    let start_time = Instant::now();
    let mut last_output = Tensor::zeros((seq_len, d_model), candle_core::DType::F32, &device)?;

    for _ in 0..iterations {
        last_output = model.forward(&input_tensor)?;
    }

    let elapsed = start_time.elapsed();
    let total_particles_processed = seq_len * num_shards * iterations;
    let particles_per_sec = total_particles_processed as f64 / elapsed.as_secs_f64();

    println!("\nOutput Sequence Tensor Shape: {:?}", last_output.shape());

    if benchmark {
        println!("\n=== ANNP High-Throughput Performance Benchmark ===");
        println!("Iterations Executed: {}", iterations);
        println!(
            "Total Processing Time: {:.4} seconds",
            elapsed.as_secs_f64()
        );
        println!("Particles Processed: {}", total_particles_processed);
        println!(
            "Particle Throughput: {:.2} particles/sec",
            particles_per_sec
        );
        println!(
            "Average Latency per Pass: {:.4} ms",
            (elapsed.as_secs_f64() * 1000.0) / iterations as f64
        );
        println!(
            "Memory Overhead per Node: ~{:.2} KB",
            (model.config.d_head * 4 * 16) as f64 / 1024.0
        );
    }

    Ok(())
}
