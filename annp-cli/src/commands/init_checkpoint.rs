use crate::checkpoint::ModelCheckpoint;
use crate::commands::train::select_device;
use crate::config::AnnpTomlConfig;
use annp_model::ANNPModel;
use clap::Args;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct InitCheckpointArgs {
    /// Path to TOML configuration file
    #[arg(short = 'c', long = "config", default_value = "annp_config.toml")]
    pub config: PathBuf,

    /// Destination path for initialized model checkpoint (e.g. "checkpoints/init_k4.annpb")
    #[arg(short = 'o', long = "output")]
    pub output: PathBuf,

    /// Checkpoint format ("annpb" or "json") (default: auto from file extension or annpb)
    #[arg(short = 'f', long = "format")]
    pub format: Option<String>,

    /// Override k_neighbors topology connectivity degree (e.g. 2, 4, 8)
    #[arg(short = 'k', long = "k-neighbors")]
    pub k_neighbors: Option<usize>,

    /// Target compute device ("cpu", "cuda", "auto")
    #[arg(short = 'd', long = "device", default_value = "cpu")]
    pub device: String,

    /// Optional random seed for reproducible deterministic initialization
    #[arg(short = 's', long = "seed")]
    pub seed: Option<u64>,
}

pub fn execute_init_checkpoint(args: InitCheckpointArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading configuration from: {:?}", args.config);
    let mut toml_config = AnnpTomlConfig::load_from_file(&args.config)?;

    if let Some(k) = args.k_neighbors {
        println!(
            "Overriding k_neighbors connectivity: {:?} -> {}",
            toml_config.model.k_neighbors, k
        );
        toml_config.model.k_neighbors = Some(k);
    }

    let core_config = toml_config.to_core_config();
    let (device, use_cuda) = select_device(&args.device);

    let num_shards = core_config.num_shards;
    let d_model = core_config.d_model();

    println!(
        "Initializing fresh ANNP model: {} nodes ({}x{}), d_head={}, d_model={}, k_neighbors={} (Seed: {:?})",
        core_config.num_nodes(),
        core_config.mesh_rows,
        core_config.mesh_cols,
        core_config.d_head,
        d_model,
        core_config.k_neighbors,
        args.seed
    );

    let model = if let Some(seed) = args.seed {
        ANNPModel::new_with_seed(
            core_config.num_nodes(),
            num_shards,
            core_config,
            device.clone(),
            use_cuda,
            seed,
        )
    } else {
        ANNPModel::new_with_cuda(
            core_config.num_nodes(),
            num_shards,
            core_config,
            device.clone(),
            use_cuda,
        )
    };

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let ckpt = ModelCheckpoint::extract_from_model(&model, 0, 0);

    let use_binary = args
        .format
        .map(|f| f.to_lowercase() == "annpb" || f.to_lowercase() == "bin")
        .unwrap_or_else(|| {
            args.output
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.to_lowercase() == "annpb" || ext.to_lowercase() == "bin")
                .unwrap_or(true)
        });

    if use_binary {
        ckpt.save_binary(&args.output)?;
    } else {
        ckpt.save(&args.output)?;
    }

    let file_size = fs::metadata(&args.output)?.len();
    println!(
        "Successfully exported initialized checkpoint to {:?} ({:.2} KB, format: {})",
        args.output,
        file_size as f64 / 1024.0,
        if use_binary {
            "ANNPB (Binary v11)"
        } else {
            "JSON"
        }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_checkpoint_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir = std::env::temp_dir().join(format!("annp_test_init_{}", std::process::id()));
        fs::create_dir_all(&temp_dir)?;
        let config_path = temp_dir.join("test_config.toml");
        fs::write(
            &config_path,
            r#"
[model]
mesh_rows = 2
mesh_cols = 2
num_shards = 2
d_head = 8
ffn_expansion = 2
initial_energy = 1.0
max_hop = 4
min_hop = 1
k_neighbors = 4

[eviction]
subnode_max = 8

[train]
epochs = 1
learning_rate = 0.001
"#,
        )?;
        let out_path1 = temp_dir.join("init1.annpb");
        let out_path2 = temp_dir.join("init2.annpb");

        let args1 = InitCheckpointArgs {
            config: config_path.clone(),
            output: out_path1.clone(),
            format: Some("annpb".into()),
            k_neighbors: Some(4),
            device: "cpu".into(),
            seed: Some(4242),
        };

        let args2 = InitCheckpointArgs {
            config: config_path.clone(),
            output: out_path2.clone(),
            format: Some("annpb".into()),
            k_neighbors: Some(4),
            device: "cpu".into(),
            seed: Some(4242),
        };

        execute_init_checkpoint(args1)?;
        execute_init_checkpoint(args2)?;

        let bytes1 = fs::read(&out_path1)?;
        let bytes2 = fs::read(&out_path2)?;

        assert_eq!(
            bytes1, bytes2,
            "Seeded init checkpoints must be byte-identical"
        );

        let ckpt1 = ModelCheckpoint::load(&out_path1)?;
        assert_eq!(ckpt1.config.k_neighbors, 4);

        let _ = fs::remove_dir_all(temp_dir);
        Ok(())
    }
}
