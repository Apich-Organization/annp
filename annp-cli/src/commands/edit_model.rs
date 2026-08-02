use crate::checkpoint::ModelCheckpoint;
use crate::config::AnnpTomlConfig;
use annp_core::OnlineStats;
use rand::Rng;
use std::fs;
use std::path::PathBuf;

pub fn execute_edit_model(
    checkpoint_path: PathBuf,
    config_path: Option<PathBuf>,
    max_hop: Option<u16>,
    min_hop: Option<u16>,
    initial_energy: Option<f32>,
    weight_decay: Option<f32>,
    epoch: Option<usize>,
    stage: Option<usize>,
    reset_state: bool,
    reset_stats: bool,
    reset_fast_weights: bool,
    reset_routing: bool,
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
        "Loaded Checkpoint: {:?} (Stage {}, Epoch {}, Nodes: {})",
        checkpoint_path,
        ckpt.stage_completed,
        ckpt.epoch_completed,
        ckpt.nodes.len()
    );

    // 3. Update from TOML if provided
    if let Some(ref cfg_p) = config_path {
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
    if let Some(val) = initial_energy {
        println!(
            "  - Override initial_energy: {} -> {}",
            ckpt.config.initial_energy, val
        );
        ckpt.config.initial_energy = val;
    }
    if let Some(val) = weight_decay {
        println!(
            "  - Override weight_decay: {} -> {}",
            ckpt.config.weight_decay, val
        );
        ckpt.config.weight_decay = val;
    }

    // 5. Stage / Epoch Overrides
    if let Some(e) = epoch {
        println!(
            "  - Override epoch_completed: {} -> {}",
            ckpt.epoch_completed, e
        );
        ckpt.epoch_completed = e;
    }
    if let Some(s) = stage {
        println!(
            "  - Override stage_completed: {} -> {}",
            ckpt.stage_completed, s
        );
        ckpt.stage_completed = s;
    }

    // 6. State Resets
    if reset_state {
        println!(
            "  - Resetting transient TD state (last_p_in, last_prediction, last_token_id) for all nodes"
        );
        for node in &mut ckpt.nodes {
            node.last_p_in.clear();
            node.last_prediction.clear();
            node.last_token_id = None;
        }
    }

    if reset_stats {
        println!(
            "  - Resetting runtime statistics (activation counts, credit stats, health base) for all nodes/subnodes"
        );
        for node in &mut ckpt.nodes {
            node.activation_count = 0;
            node.cumulative_sequence_len = 0;
            for subnode in &mut node.subnodes {
                subnode.activation_count = 0;
                subnode.credit_stats = OnlineStats::default();
                subnode.health = ckpt.config.health_base;
            }
        }
    }

    if reset_fast_weights {
        println!("  - Resetting associative fast_weight memory matrices and cumulative energy");
        for node in &mut ckpt.nodes {
            node.fast_weight.fill(0.0);
            node.cumulative_energy = ckpt.config.initial_energy;
        }
    }

    if reset_routing {
        println!("  - Resetting P2P routing tables (weights & edge credits)");
        let mut rng = rand::rng();
        let d_head = ckpt.config.d_head;
        let scale = (1.0 / (d_head as f32)).sqrt();

        for rt in &mut ckpt.routing_tables {
            let num_neighbors = rt.neighbors.len();
            rt.weights = (0..d_head * num_neighbors)
                .map(|_| rng.random_range(-scale..scale))
                .collect();
            rt.edge_credit = vec![OnlineStats::default(); num_neighbors];
        }
    }

    // 7. Save updated checkpoint
    ckpt.save(&checkpoint_path)?;
    println!("Successfully updated checkpoint: {:?}", checkpoint_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_core::MicroBlockConfig;
    use annp_model::ANNPModel;
    use candle_core::Device;

    fn create_test_checkpoint() -> ModelCheckpoint {
        let config = MicroBlockConfig {
            mesh_rows: 2,
            mesh_cols: 2,
            d_head: 16,
            max_hop: 8,
            min_hop: 2,
            initial_energy: 10.0,
            weight_decay: 0.0001,
            health_base: 1.0,
            ..MicroBlockConfig::default()
        };
        let model = ANNPModel::new(4, 2, config, Device::Cpu);
        ModelCheckpoint::extract_from_model(&model, 1, 3)
    }

    #[test]
    fn test_edit_model_config_and_stage_overrides() {
        let tmp_dir = std::env::temp_dir();
        let ckpt_path = tmp_dir.join("test_edit_model_override.annpb");
        let bak_path = tmp_dir.join("test_edit_model_override.annpb.bak");

        let ckpt = create_test_checkpoint();
        ckpt.save(&ckpt_path).unwrap();

        let res = execute_edit_model(
            ckpt_path.clone(),
            None,
            Some(16),   // max_hop
            Some(4),    // min_hop
            Some(20.0), // initial_energy
            Some(0.01), // weight_decay
            Some(10),   // epoch
            Some(3),    // stage
            false,
            false,
            false,
            false,
        );
        assert!(res.is_ok());

        // Verify backup exists and has old stage/epoch
        assert!(bak_path.exists());
        let bak_ckpt = ModelCheckpoint::load(&bak_path).unwrap();
        assert_eq!(bak_ckpt.stage_completed, 1);
        assert_eq!(bak_ckpt.epoch_completed, 3);
        assert_eq!(bak_ckpt.config.max_hop, 8);

        // Verify loaded updated checkpoint
        let updated = ModelCheckpoint::load(&ckpt_path).unwrap();
        assert_eq!(updated.stage_completed, 3);
        assert_eq!(updated.epoch_completed, 10);
        assert_eq!(updated.config.max_hop, 16);
        assert_eq!(updated.config.min_hop, 4);
        assert!((updated.config.initial_energy - 20.0).abs() < 1e-6);
        assert!((updated.config.weight_decay - 0.01).abs() < 1e-6);

        let _ = fs::remove_file(ckpt_path);
        let _ = fs::remove_file(bak_path);
    }

    #[test]
    fn test_edit_model_resets() {
        let tmp_dir = std::env::temp_dir();
        let ckpt_path = tmp_dir.join("test_edit_model_resets.annpb");
        let bak_path = tmp_dir.join("test_edit_model_resets.annpb.bak");

        let mut ckpt = create_test_checkpoint();
        // Artificially dirty state
        for node in &mut ckpt.nodes {
            node.last_p_in = vec![1.0; 16];
            node.last_prediction = vec![2.0; 16];
            node.last_token_id = Some(999);
            node.activation_count = 50;
            node.cumulative_energy = 55.5;
            node.fast_weight = vec![0.5; 16 * 16];
            for subnode in &mut node.subnodes {
                subnode.activation_count = 50;
                subnode.health = 0.42;
                subnode.credit_stats.observe(1.23);
            }
        }
        for rt in &mut ckpt.routing_tables {
            for stats in &mut rt.edge_credit {
                stats.observe(0.88);
            }
        }
        ckpt.save(&ckpt_path).unwrap();

        let res = execute_edit_model(
            ckpt_path.clone(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            true, // reset_state
            true, // reset_stats
            true, // reset_fast_weights
            true, // reset_routing
        );
        assert!(res.is_ok());

        let updated = ModelCheckpoint::load(&ckpt_path).unwrap();
        for node in &updated.nodes {
            // State reset verification
            assert!(node.last_p_in.is_empty());
            assert!(node.last_prediction.is_empty());
            assert!(node.last_token_id.is_none());

            // Stats reset verification
            assert_eq!(node.activation_count, 0);
            assert_eq!(node.cumulative_sequence_len, 0);
            for subnode in &node.subnodes {
                assert_eq!(subnode.activation_count, 0);
                assert_eq!(subnode.credit_stats.count, 0.0);
                assert!((subnode.health - updated.config.health_base).abs() < 1e-6);
            }

            // Fast weight reset verification
            assert!(node.fast_weight.iter().all(|&x| x == 0.0));
            assert!((node.cumulative_energy - updated.config.initial_energy).abs() < 1e-6);
        }

        // Routing reset verification
        for rt in &updated.routing_tables {
            for stats in &rt.edge_credit {
                assert_eq!(stats.count, 0.0);
                assert_eq!(stats.mean, 0.0);
            }
        }

        let _ = fs::remove_file(ckpt_path);
        let _ = fs::remove_file(bak_path);
    }
}
