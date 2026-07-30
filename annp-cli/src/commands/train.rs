use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use crate::dataset::{DatasetFormat, DatasetStream};
use annp_model::ANNPModel;
use annp_trainer::trainer::Trainer;
use candle_core::Device;
use std::path::PathBuf;

pub fn select_device(device_str: &str) -> (Device, bool) {
    let dev_str = device_str.to_lowercase();
    let want_cpu = dev_str == "cpu";
    let want_cuda = matches!(dev_str.as_str(), "cuda" | "gpu" | "auto" | "");

    let cuda_available = annp_cuda::is_cuda_available();
    let use_cuda = !want_cpu && want_cuda && cuda_available;

    let device = if use_cuda {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    if use_cuda {
        println!("Selected Compute Device: Cuda(0) [ANNP Native CUDA GPU Acceleration Engine]");
    } else {
        if !want_cpu && want_cuda && !cuda_available {
            println!(
                "Notice: CUDA requested but CUDA acceleration is unavailable on this system. Falling back to CPU."
            );
        }
        println!("Selected Compute Device: Cpu [AVX2 SIMD Engine]");
    }

    (device, use_cuda)
}

pub fn execute_train(
    config_path: PathBuf,
    resume_from: Option<PathBuf>,
    checkpoint_format: String,
    device_target: String,
    output_dir: PathBuf,
    log_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let logger = crate::logger::AnnpLogger::new(&log_dir, "train", None);

    logger.log(
        "INIT",
        &format!("Loading ANNP Configuration from: {:?}", config_path),
    );
    let toml_config = AnnpTomlConfig::load_from_file(config_path)?;
    let core_config = toml_config.to_core_config();

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
    model.is_training = true;

    let mut start_epoch = 0;

    // Handle Checkpoint Resume
    let is_resumed = resume_from.is_some();
    if let Some(ref ckpt_path) = resume_from {
        logger.log(
            "RESUME",
            &format!("Resuming model state from checkpoint: {:?}", ckpt_path),
        );
        let ckpt = ModelCheckpoint::load(ckpt_path)?;
        ckpt.apply_to_model(&mut model);
        start_epoch = ckpt.epoch_completed + 1;
        logger.log(
            "RESUME",
            &format!("Successfully resumed from Epoch {}.", start_epoch),
        );
    }

    logger.log(
        "SYSTEM",
        "=== Starting ANNP Evolutionary Training Pipeline ===",
    );
    logger.log(
        "SYSTEM",
        &format!(
            "Total Micro-Block Nodes: {} ({}x{}) | Particle d_head: {} | Total d_model: {}",
            model.num_nodes,
            model.config.mesh_rows,
            model.config.mesh_cols,
            model.config.d_head,
            d_model
        ),
    );

    let use_binary =
        checkpoint_format.to_lowercase() == "annpb" || checkpoint_format.to_lowercase() == "binary";
    let file_ext = if use_binary { "annpb" } else { "json" };

    let train_epochs = toml_config.train.epochs;
    let train_lr = toml_config.train.learning_rate;
    let train_cfg = &toml_config.train;

    logger.log(
        "TRAIN",
        &format!(
            ">>> Launching Training: Epochs = {}, LR = {}",
            train_epochs, train_lr
        ),
    );

    let dataset_path = train_cfg.dataset_path.as_deref().unwrap_or("synthetic");
    let format_str = train_cfg.dataset_format.as_deref().unwrap_or("synthetic");
    let dataset_fmt = DatasetFormat::parse(format_str);

    let epoch_start_val = if is_resumed { start_epoch } else { 0 };

    if is_resumed && epoch_start_val > 0 {
        logger.log(
            "RESUME",
            &format!(
                "Resuming Training: Skipping completed Epochs 1..{}, starting directly at Epoch {}/{}",
                epoch_start_val, epoch_start_val + 1, train_epochs
            ),
        );
    }

    for epoch in epoch_start_val..train_epochs {
        logger.log(
            "EPOCH",
            &format!(
                "Streaming dataset from: {} ({}) [Epoch {}/{}]",
                dataset_path,
                format_str,
                epoch + 1,
                train_epochs
            ),
        );

        let stream = DatasetStream::new(dataset_path, dataset_fmt, d_model, &device)?;
        let mut epoch_loss_sum = 0.0f32;
        let mut step_count = 0;
        let mut rolling_ema = 0.0f32;

        for (batch_idx, res) in stream.enumerate() {
            let tensor = res?;
            let mut trainer = Trainer::new(train_lr);
            let step_loss = trainer.train_step_with_epoch(&mut model, &tensor, epoch)?;

            epoch_loss_sum += step_loss;
            step_count += 1;

            rolling_ema = if step_count == 1 {
                step_loss
            } else {
                0.9f32 * rolling_ema + 0.1f32 * step_loss
            };

            if (batch_idx + 1) % 2 == 0 {
                logger.log_step(
                    epoch + 1,
                    train_epochs,
                    batch_idx + 1,
                    step_loss,
                    rolling_ema,
                );
            }
        }

        let avg_epoch_loss = epoch_loss_sum / step_count.max(1) as f32;
        logger.log(
            "EPOCH_END",
            &format!(
                "Epoch {}/{} Complete. Average Loss: {:.6}",
                epoch + 1,
                train_epochs,
                avg_epoch_loss
            ),
        );

        // Save intermediate checkpoint (.annpb binary or .json)
        let ckpt = ModelCheckpoint::extract_from_model(&model, 0, epoch);
        let ckpt_filename = output_dir.join(format!("checkpoint_epoch{}.{}", epoch + 1, file_ext));
        ckpt.save(&ckpt_filename)?;
        println!("Saved intermediate checkpoint to: {:?}", ckpt_filename);
    }

    println!("\n============================================================");
    println!("ANNP Training Completed Successfully!");
    println!(
        "Final Model Checkpoints saved to directory: {:?}",
        output_dir
    );
    println!("============================================================");

    Ok(())
}
