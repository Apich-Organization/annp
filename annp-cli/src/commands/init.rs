use crate::config::AnnpTomlConfig;
use std::path::PathBuf;

pub fn execute_init(output: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let config = AnnpTomlConfig::default();
    config.save_to_file(&output)?;
    println!(
        "Successfully initialized ANNP configuration TOML file at: {:?}",
        output
    );
    println!(
        "Edit this configuration to customize Micro-Block dimensions, 4-stage datasets, and eviction parameters."
    );
    Ok(())
}
