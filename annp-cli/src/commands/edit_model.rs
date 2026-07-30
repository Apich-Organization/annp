use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use std::fs;
use std::path::PathBuf;

pub fn execute_edit_model(
    checkpoint_path: PathBuf,
    config_path: Option<PathBuf>,
    max_hop: Option<u16>,
    min_hop: Option<u16>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !checkpoint_path.exists() {
        return Err(format!("Checkpoint file {:?} does not exist", checkpoint_path).into());
    }

    // 1. Automatic Backup: checkpoint.annpb -> checkpoint.annpb.bak
    let ext = checkpoint_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("annpb");
    let backup_path = checkpoint_path.with_extension(format!("{}.bak", ext));

    fs::copy(&checkpoint_path, &backup_path)?;
    println!("Created automatic checkpoint backup: {:?}", backup_path);

    // 2. Load Checkpoint
    let mut ckpt = ModelCheckpoint::load(&checkpoint_path)?;
    println!(
        "Loaded Checkpoint: {:?} (Completed Stage {}, Epoch {})",
        checkpoint_path, ckpt.stage_completed, ckpt.epoch_completed
    );

    // 3. Update from TOML if provided
    if let Some(cfg_p) = config_path {
        println!("Applying configuration settings from TOML: {:?}", cfg_p);
        let toml_cfg = AnnpTomlConfig::load_from_file(cfg_p)?;
        ckpt.config = toml_cfg.to_core_config();
    }

    // 4. Individual CLI Overrides
    if let Some(val) = max_hop {
        println!("  - Override max_hop: {} -> {}", ckpt.config.max_hop, val);
        ckpt.config.max_hop = val;
    }
    if let Some(val) = min_hop {
        println!("  - Override min_hop: {} -> {}", ckpt.config.min_hop, val);
        ckpt.config.min_hop = val;
    }


    // 5. Save updated checkpoint
    ckpt.save(&checkpoint_path)?;
    println!(
        "Successfully updated embedded configuration in checkpoint: {:?}",
        checkpoint_path
    );

    Ok(())
}
