use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use crate::tokenizer::AnnpTokenizer;
use annp_model::ANNPModel;
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
    log_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let logger = crate::logger::AnnpLogger::new(&log_dir, "run", None);
    logger.log(
        "INIT",
        &format!("Loading ANNP Configuration from: {:?}", config_path),
    );
    let toml_config = AnnpTomlConfig::load_from_file(config_path)?;
    let core_config = toml_config.to_core_config();

    if let Some(tau) = temperature_override {
        println!(
            "Overriding Routing Temperature: tau = {:.3} (ignored, parameter removed)",
            tau
        );
    }

    let (device, use_cuda) = select_device(&device_target);

    let num_shards = core_config.num_shards;
    let d_model = core_config.d_model();
    let mut model = ANNPModel::new_with_cuda(
        core_config.num_nodes(),
        num_shards,
        core_config,
        device.clone(),
        use_cuda,
    );
    model.is_training = continual_mode;

    if let Some(ckpt_path) = checkpoint_path {
        println!("Loading Checkpoint: {:?}", ckpt_path);
        let ckpt = ModelCheckpoint::load(&ckpt_path)?;
        ckpt.apply_to_model(&mut model);
        println!("Successfully loaded weights from {:?}", ckpt_path);
    }

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

    let is_random_mode = input_text.is_none();
    println!(
        "\n=== Executing ANNP Model Inference Pass ({}) ===",
        run_mode_str
    );
    if is_random_mode {
        println!("Mode: Random Initial Tokens");
    } else {
        println!("Mode: Prompt Auto-Regressive Generation");
    }

    println!("Input Tensor Shape: {:?}", input_tensor.shape());

    let generate_len = if benchmark { 5 } else { 20 };
    let start_time = Instant::now();
    let mut current_sequence = input_tensor.clone();
    let mut total_particles_processed = 0;

    // Store decoded text
    let mut generated_ids = Vec::new();

    // For stateful autoregressive generation, we only feed the new token each step.
    let mut next_token_tensor = input_tensor.clone();

    for (current_len, _step) in (seq_len..).zip(0..generate_len) {
        // Feed only the new token, using current_len - seq_len (or just the step index) for offset if needed,
        // but since we want the actual sequence index in the particle, we pass current_len - next_token_tensor.dim(0).
        let batch_len = next_token_tensor.dim(0)?;

        let (out, _) = model.forward(&next_token_tensor, current_sequence.dim(0)?, None)?;
        total_particles_processed += batch_len * num_shards;

        if continual_mode {
            // Continual adaptation hook - model.forward automatically updates weights when is_training is true
        }

        // Extract the prediction for the next token (the output of the last sequence element of the newly processed batch)
        let flat_out = out.flatten_all()?.to_vec1::<f32>()?;
        let last_out = &flat_out[(batch_len - 1) * d_model..batch_len * d_model];

        // Decode via Nearest Neighbor Search over vocab (1..32000)
        let mut best_id = 1u32;
        let mut best_score = -f32::INFINITY;

        let target_pos = current_len;

        for token_id in 1..32000u32 {
            // Reconstruct the expected activation tensor vector for this token_id at target_pos
            let mut expected_vec = Vec::with_capacity(d_model);
            let mut seed = (token_id as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(1);

            for d in 0..d_model {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let rand_f32 = ((seed & 0xFFFFFFFF) as f32 / 4294967295.0f32) * 2.0f32 - 1.0f32;
                let pos_enc = (target_pos as f32 * 0.05f32 + d as f32 * 0.01f32).sin() * 0.1f32;
                expected_vec.push(rand_f32 + pos_enc);
            }
            let rms = (expected_vec.iter().map(|v| v * v).sum::<f32>() / d_model as f32)
                .sqrt()
                .max(1e-6);

            // Compute cosine similarity
            let mut dot = 0.0;
            for d in 0..d_model {
                dot += last_out[d] * (expected_vec[d] / rms);
            }
            if dot > best_score {
                best_score = dot;
                best_id = token_id;
            }
        }

        generated_ids.push(best_id);

        // Append the new token to the sequence
        let new_token_text = tokenizer.decode(&[best_id]);
        print!("{} ", new_token_text);
        std::io::stdout().flush().unwrap();

        // Re-encode the newly generated token and append to current_sequence
        // Since encode_to_tensor expects a full string and we want it to be at target_pos,
        // we can just use our reconstructed logic to build the tensor directly
        let mut next_vec = Vec::with_capacity(d_model);
        let mut seed = (best_id as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(1);
        for d in 0..d_model {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let rand_f32 = ((seed & 0xFFFFFFFF) as f32 / 4294967295.0f32) * 2.0f32 - 1.0f32;
            let pos_enc = (target_pos as f32 * 0.05f32 + d as f32 * 0.01f32).sin() * 0.1f32;
            next_vec.push(rand_f32 + pos_enc);
        }
        let rms = (next_vec.iter().map(|v| v * v).sum::<f32>() / d_model as f32)
            .sqrt()
            .max(1e-6);
        for v in next_vec.iter_mut() {
            *v /= rms;
        }

        let next_tensor = Tensor::from_vec(next_vec, (1, d_model), &device)?;
        current_sequence = Tensor::cat(&[&current_sequence, &next_tensor], 0)?;
        next_token_tensor = next_tensor;
    }

    println!(
        "\n\nFinal Full Generated Sequence: \"{} {}\"",
        input_text.unwrap_or_else(|| "<random_init>".to_string()),
        tokenizer.decode(&generated_ids)
    );

    if continual_mode {
        println!(
            "Online Continual Learning Pass applied to all {} Micro-Block nodes.",
            model.num_nodes
        );
    }

    // Save binary output tensor if requested (.annpb)
    if let Some(save_path) = save_output {
        println!("Saving output tensor to binary file: {:?}", save_path);
        let flat_output = current_sequence.flatten_all()?.to_vec1::<f32>()?;
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
        let elapsed = start_time.elapsed();
        let particles_per_sec = total_particles_processed as f64 / elapsed.as_secs_f64();

        println!("\n=== ANNP High-Throughput Performance Benchmark ===");
        println!("Mode: {}", run_mode_str);
        println!("Iterations Executed: {}", generate_len);
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
            (elapsed.as_secs_f64() * 1000.0) / generate_len as f64
        );
        println!(
            "Memory Overhead per Node: ~{:.2} KB",
            (model.config.d_head * 4 * 16) as f64 / 1024.0
        );
    }

    Ok(())
}
