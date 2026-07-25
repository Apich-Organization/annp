use crate::checkpoint::ModelCheckpoint;
use std::fs;
use std::path::PathBuf;

pub fn execute_export(
    checkpoint_path: PathBuf,
    topology_out: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Checkpoint from: {:?}", checkpoint_path);
    let ckpt = ModelCheckpoint::load(checkpoint_path)?;

    let json_content = serde_json::to_string_pretty(&ckpt.routing_tables)?;
    if let Some(parent) = topology_out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&topology_out, json_content)?;

    println!("Successfully exported P2P Routing Topology for {} nodes to: {:?}", ckpt.routing_tables.len(), topology_out);
    Ok(())
}
