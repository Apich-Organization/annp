use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use crate::tokenizer::AnnpTokenizer;
use annp_model::ANNPModel;
use annp_trainer::Stage1HardeningTrainer;
use candle_core::Tensor;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use super::train::select_device;

pub fn execute_run(
    config_path: PathBuf,
    checkpoint_path: Option<PathBuf>,
    input_text: Option<String>,
    temperature_override: Option<f32>,
    device_target: String,
    continual_mode: bool,
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

    let (device, use_cuda) = select_device(&device_target);

    let num_shards = 4;
    let mut model = ANNPModel::new_with_cuda(
        core_config.num_nodes(),
        num_shards,
        core_config,
        device.clone(),
        use_cuda,
    );

    if let Some(ckpt_path) = checkpoint_path {
        println!("Loading Checkpoint: {:?}", ckpt_path);
        let ckpt = ModelCheckpoint::load(&ckpt_path)?;
        ckpt.apply_to_model(&mut model);
        println!("Successfully loaded weights from {:?}", ckpt_path);
    }

    let d_model = num_shards * model.config.d_head;

    // Load Hugging Face / SentencePiece Tokenizer from tokenizer.model
    let tokenizer = AnnpTokenizer::load_from_file("tokenizer.model");

    // Parse input text using AnnpTokenizer or generate random input
    let (input_tensor, seq_len) = if let Some(ref prompt) = input_text {
        println!("Parsing input prompt with AnnpTokenizer: \"{}\"", prompt);
        let ids = tokenizer.encode(prompt);
        println!("Encoded Token IDs (Count {}): {:?}", ids.len(), ids);
        let tensor = tokenizer.encode_to_tensor(prompt, d_model, &device)?;
        let s_len = tensor.dim(0)?;
        (tensor, s_len)
    } else {
        let default_seq_len = 16;
        (
            Tensor::randn(0.0f32, 1.0f32, (default_seq_len, d_model), &device)?,
            default_seq_len,
        )
    };

    let run_mode_str = if continual_mode {
        "Continual Online Adaptation Mode (Dynamic Hardening)"
    } else {
        "Static Production Mode (Deterministic Frozen Weights)"
    };

    println!(
        "\n=== Executing ANNP Model Inference Pass ({}) ===",
        run_mode_str
    );
    println!("Input Tensor Shape: {:?}", input_tensor.shape());

    let iterations = if benchmark { 50 } else { 1 };
    let start_time = Instant::now();
    let mut last_output = Tensor::zeros((seq_len, d_model), candle_core::DType::F32, &device)?;

    for _ in 0..iterations {
        last_output = model.forward(&input_tensor)?;

        if continual_mode {
            // Apply Online Plasticity Hardening & Continual Adaptation
            let trainer = Stage1HardeningTrainer::new(0.02, 0.001, 1.5);
            trainer.apply_plastic_hardening(&mut model);
        }
    }

    let elapsed = start_time.elapsed();
    let total_particles_processed = seq_len * num_shards * iterations;
    let particles_per_sec = total_particles_processed as f64 / elapsed.as_secs_f64();

    println!("Output Sequence Tensor Shape: {:?}", last_output.shape());

    if input_text.is_some() {
        let output_vec = last_output.flatten_all()?.to_vec1::<f32>()?;
        let mut decoded_ids = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let row_slice = &output_vec[t * d_model..(t + 1) * d_model];
            let mean_val: f32 = row_slice.iter().sum::<f32>() / d_model as f32;
            let token_id = (mean_val.abs() * 1000.0) as u32;
            decoded_ids.push(token_id);
        }
        let decoded_text = tokenizer.decode(&decoded_ids);
        println!("Decoded ANNP Output Sequence Text: \"{}\"", decoded_text);
    }

    if continual_mode {
        println!(
            "Online Continual Learning Pass applied to all {} Micro-Block nodes.",
            model.num_nodes
        );
    }

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
        println!("Mode: {}", run_mode_str);
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
