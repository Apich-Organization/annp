use annp_core::MicroBlockConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Fully commented TOML configuration file representation for ANNP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnpTomlConfig {
    pub model: ModelSection,
    pub eviction: EvictionSection,
    pub train: TrainConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSection {
    pub num_shards: Option<usize>,
    pub mesh_rows: usize,
    pub mesh_cols: usize,
    pub d_head: usize,
    pub ffn_expansion: usize,
    pub initial_energy: f32,
    pub max_hop: u16,
    pub min_hop: u16,
    pub norm_strategy: String, // "RMSNorm" or "SphereNorm"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionSection {
    pub subnode_max: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub dataset_path: Option<String>,
    pub dataset_format: Option<String>,
}

impl Default for AnnpTomlConfig {
    fn default() -> Self {
        Self {
            model: ModelSection {
                num_shards: Some(4),
                mesh_rows: 10,
                mesh_cols: 10,
                d_head: 64,
                ffn_expansion: 8,
                initial_energy: 1.0,
                max_hop: 100,
                min_hop: 3,
                norm_strategy: "RMSNorm".to_string(),
            },
            eviction: EvictionSection {
                subnode_max: Some(8),
            },
            train: TrainConfig {
                epochs: 8,
                learning_rate: 0.02,
                dataset_path: Some("synthetic".to_string()),
                dataset_format: Some("synthetic".to_string()),
            },
        }
    }
}

impl AnnpTomlConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn to_core_config(&self) -> MicroBlockConfig {
        MicroBlockConfig {
            num_shards: self.model.num_shards.unwrap_or(4),
            mesh_rows: self.model.mesh_rows,
            mesh_cols: self.model.mesh_cols,
            d_head: self.model.d_head,
            ffn_expansion: self.model.ffn_expansion,
            initial_energy: self.model.initial_energy,
            max_hop: self.model.max_hop,
            min_hop: self.model.min_hop,
            subnode_max: self.eviction.subnode_max.unwrap_or(8),
        }
    }
}
