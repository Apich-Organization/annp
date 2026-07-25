use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use crate::dataset::{load_dataset, DatasetFormat};
use annp_model::ANNPModel;
use annp_trainer::{
    Stage0WaveTrainer, Stage1RouterTrainer, Stage2PonderTrainer, Stage3ContinualTrainer,
};
use candle_core::Device;
use std::path::PathBuf;

pub fn execute_train(
    config_path: PathBuf,
    stage_target: String,
    resume_from: Option<PathBuf>,
    output_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading ANNP Configuration from: {:?}", config_path);
    let toml_config = AnnpTomlConfig::load_from_file(config_path)?;
    let core_config = toml_config.to_core_config();

    let device = Device::Cpu;
    let num_shards = 4;
    let mut model = ANNPModel::new(core_config.num_nodes(), num_shards, core_config, device.clone());

    let mut start_stage = 0;
    let mut start_epoch = 0;

    // Handle Checkpoint Resume
    if let Some(ckpt_path) = resume_from {
        println!("Resuming model state from checkpoint: {:?}", ckpt_path);
        let ckpt = ModelCheckpoint::load(ckpt_path)?;
        ckpt.apply_to_model(&mut model);
        start_stage = ckpt.stage_completed;
        start_epoch = ckpt.epoch_completed + 1;
        println!("Successfully resumed from Stage {}, Epoch {}.", start_stage, start_epoch);
    }

    let d_model = num_shards * model.config.d_head;

    // Stage Selector
    let stages_to_run: Vec<usize> = match stage_target.to_lowercase().as_str() {
        "0" | "wave" => vec![0],
        "1" | "router" => vec![1],
        "2" | "ponder" => vec![2],
        "3" | "continual" => vec![3],
        _ => vec![0, 1, 2, 3],
    };

    println!("\n=== Starting ANNP Evolutionary Training Pipeline ===");
    println!("Total Micro-Block Nodes: {} ({}x{})", model.num_nodes, model.config.mesh_rows, model.config.mesh_cols);
    println!("Particle d_head: {}, Total d_model: {}", model.config.d_head, d_model);

    for &stg in &stages_to_run {
        if stg < start_stage {
            continue;
        }

        let stage_cfg = match stg {
            0 => &toml_config.stage0_wave,
            1 => &toml_config.stage1_router,
            2 => &toml_config.stage2_ponder,
            _ => &toml_config.stage3_continual,
        };

        if !stage_cfg.enabled {
            println!("\n[Stage {}] Skipped (disabled in config).", stg);
            continue;
        }

        println!("\n------------------------------------------------------------");
        println!(">>> Launching Stage {}: Epochs = {}, LR = {}", stg, stage_cfg.epochs, stage_cfg.learning_rate);
        println!("------------------------------------------------------------");

        // Load stage-specific dataset
        let dataset_path = stage_cfg.dataset_path.as_deref().unwrap_or("synthetic");
        let format_str = stage_cfg.dataset_format.as_deref().unwrap_or("json");
        let dataset_fmt = DatasetFormat::parse(format_str);

        println!("Loading Stage {} dataset from: {} ({})", stg, dataset_path, format_str);
        let data_tensors = load_dataset(dataset_path, dataset_fmt, d_model, &device)?;
        println!("Loaded {} dataset batches.", data_tensors.len());

        let epoch_start_val = if stg == start_stage { start_epoch } else { 0 };

        for epoch in epoch_start_val..stage_cfg.epochs {
            let mut epoch_loss_sum = 0.0f32;
            let mut step_count = 0;

            for (batch_idx, tensor) in data_tensors.iter().enumerate() {
                let step_loss = match stg {
                    0 => {
                        let mut trainer = Stage0WaveTrainer::new(stage_cfg.learning_rate);
                        trainer.train_step(&mut model, tensor)?
                    }
                    1 => {
                        let mut trainer = Stage1RouterTrainer::new(model.config.temperature, 0.05);
                        trainer.train_step(&mut model, tensor)?
                    }
                    2 => {
                        let mut trainer = Stage2PonderTrainer::new(0.01);
                        let (loss, _avg_hops) = trainer.train_step(&mut model, tensor)?;
                        loss
                    }
                    _ => {
                        let trainer = Stage3ContinualTrainer::new(stage_cfg.learning_rate, 0.001, 1.5);
                        trainer.apply_plastic_hardening(&mut model);
                        let mut trainer0 = Stage0WaveTrainer::new(stage_cfg.learning_rate);
                        trainer0.train_step(&mut model, tensor)?
                    }
                };

                epoch_loss_sum += step_loss;
                step_count += 1;

                if (batch_idx + 1) % 1 == 0 {
                    let rolling_avg = epoch_loss_sum / step_count as f32;
                    println!(
                        "[Stage {} | Epoch {:2}/{:2} | Batch {:3}/{:3}] Step Loss: {:.6} | Rolling Loss: {:.6}",
                        stg, epoch + 1, stage_cfg.epochs, batch_idx + 1, data_tensors.len(), step_loss, rolling_avg
                    );
                }
            }

            let avg_epoch_loss = epoch_loss_sum / step_count.max(1) as f32;
            println!("==> Stage {} Epoch {}/{} Complete. Average Loss: {:.6}", stg, epoch + 1, stage_cfg.epochs, avg_epoch_loss);

            // Save intermediate checkpoint
            let ckpt = ModelCheckpoint::extract_from_model(&model, stg, epoch);
            let ckpt_filename = output_dir.join(format!("checkpoint_stage{}_epoch{}.json", stg, epoch + 1));
            ckpt.save(&ckpt_filename)?;
            println!("Saved intermediate checkpoint to: {:?}", ckpt_filename);
        }
    }

    println!("\n============================================================");
    println!("ANNP Evolutionary Training Completed Successfully!");
    println!("Final Model Checkpoints saved to directory: {:?}", output_dir);
    println!("============================================================");

    Ok(())
}
