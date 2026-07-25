use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use annp_model::ANNPModel;
use candle_core::{Device, Tensor};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

pub fn execute_run(
    config_path: PathBuf,
    checkpoint_path: Option<PathBuf>,
    input_text: Option<String>,
    temperature_override: Option<f32>,
    save_output: Option<PathBuf>,
    benchmark: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading ANNP Configuration from: {:?}", config_path);
    let toml_config = AnnpTomlConfig::load_from_file(config_path)?;
    let mut core_config = toml_config.to_core_config();

    if let Some(tau) = temperature_override {
        println!("Overriding Routing Temperature: tau = {:.3}", tau);
        core_config.temperature = tau;
    }

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
        let ckpt = ModelCheckpoint::load(&ckpt_path)?;
        ckpt.apply_to_model(&mut model);
        println!("Successfully loaded weights from {:?}", ckpt_path);
    }

    let d_model = num_shards * model.config.d_head;
    let seq_len = 16;

    // Parse input text vector or generate random input
    let input_tensor = if let Some(prompt) = input_text {
        println!("Parsing input prompt: \"{}\"", prompt);
        let mut flat = Vec::with_capacity(seq_len * d_model);
        let bytes = prompt.as_bytes();
        for t in 0..seq_len {
            for d in 0..d_model {
                let char_byte = bytes[(t * d_model + d) % bytes.len().max(1)] as f32;
                flat.push((char_byte / 255.0f32) * 2.0 - 1.0);
            }
        }
        Tensor::from_vec(flat, (seq_len, d_model), &device)?
    } else {
        Tensor::randn(0.0f32, 1.0f32, (seq_len, d_model), &device)?
    };

    println!("\n=== Executing ANNP Model Inference Pass ===");
    println!("Input Tensor Shape: {:?}", input_tensor.shape());

    let iterations = if benchmark { 50 } else { 1 };
    let start_time = Instant::now();
    let mut last_output = Tensor::zeros((seq_len, d_model), candle_core::DType::F32, &device)?;

    for _ in 0..iterations {
        last_output = model.forward(&input_tensor)?;
    }

    let elapsed = start_time.elapsed();
    let total_particles_processed = seq_len * num_shards * iterations;
    let particles_per_sec = total_particles_processed as f64 / elapsed.as_secs_f64();

    println!("Output Sequence Tensor Shape: {:?}", last_output.shape());

    // Save binary output tensor if requested (.annpb)
    if let Some(save_path) = save_output {
        println!("Saving output tensor to binary file: {:?}", save_path);
        let flat_output = last_output.flatten_all()?.to_vec1::<f32>()?;
        let mut file = File::create(&save_path)?;
        file.write_all(b"ANNPB_OUT")?;
        file.write_all(&(seq_len as u32).to_le_bytes())?;
        file.write_all(&(d_model as u32).to_le_bytes())?;

        let bytes = unsafe {
            std::slice::from_raw_parts(flat_output.as_ptr() as *const u8, flat_output.len() * 4)
        };
        file.write_all(bytes)?;
        println!(
            "Saved {} output floats to {:?}",
            flat_output.len(),
            save_path
        );
    }

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
