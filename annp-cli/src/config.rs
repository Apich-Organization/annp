use annp_core::{MicroBlockConfig, NormStrategy};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Fully commented TOML configuration file representation for ANNP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnpTomlConfig {
    pub model: ModelSection,
    pub eviction: EvictionSection,
    pub stage0_wave: StageConfig,
    pub stage1_hardening: StageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSection {
    pub mesh_rows: usize,
    pub mesh_cols: usize,
    pub d_head: usize,
    pub ffn_expansion: usize,
    pub initial_energy: f32,
    pub max_hop: u16,
    pub min_hop: u16,
    pub epsilon_p: f32,
    pub epsilon_h: f32,
    pub temperature: f32,
    pub norm_strategy: String, // "RMSNorm" or "SphereNorm"
    pub alpha_init: f32,
    pub sphere_radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionSection {
    pub lambda_temporal: f32,
    pub lambda_frequency: f32,
    pub eviction_threshold: f32,
    pub pruning_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    pub enabled: bool,
    pub epochs: usize,
    pub learning_rate: f32,
    pub dataset_path: Option<String>,
    pub dataset_format: Option<String>,
}

impl Default for AnnpTomlConfig {
    fn default() -> Self {
        Self {
            model: ModelSection {
                mesh_rows: 10,
                mesh_cols: 10,
                d_head: 64,
                ffn_expansion: 8,
                initial_energy: 1.0,
                max_hop: 100,
                min_hop: 3,
                epsilon_p: 1e-4,
                epsilon_h: 0.05,
                temperature: 1.0,
                norm_strategy: "RMSNorm".to_string(),
                alpha_init: 0.01,
                sphere_radius: 1.0,
            },
            eviction: EvictionSection {
                lambda_temporal: 0.001,
                lambda_frequency: 0.01,
                eviction_threshold: 1e-4,
                pruning_threshold: 1e-5,
            },
            stage0_wave: StageConfig {
                enabled: true,
                epochs: 8,
                learning_rate: 0.02,
                dataset_path: Some("synthetic".to_string()),
                dataset_format: Some("synthetic".to_string()),
            },
            stage1_hardening: StageConfig {
                enabled: true,
                epochs: 15,
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
        let norm_strat = if self.model.norm_strategy.to_lowercase().contains("sphere") {
            NormStrategy::SphereNormalization
        } else {
            NormStrategy::MicroRMSNorm
        };

        MicroBlockConfig {
            mesh_rows: self.model.mesh_rows,
            mesh_cols: self.model.mesh_cols,
            d_head: self.model.d_head,
            ffn_expansion: self.model.ffn_expansion,
            initial_energy: self.model.initial_energy,
            max_hop: self.model.max_hop,
            min_hop: self.model.min_hop,
            epsilon_p: self.model.epsilon_p,
            epsilon_h: self.model.epsilon_h,
            temperature: self.model.temperature,
            norm_strategy: norm_strat,
            alpha_init: self.model.alpha_init,
            sphere_radius: self.model.sphere_radius,
            lambda_temporal: self.eviction.lambda_temporal,
            lambda_frequency: self.eviction.lambda_frequency,
            eviction_threshold: self.eviction.eviction_threshold,
            pruning_threshold: self.eviction.pruning_threshold,
            queue_backpressure_alpha: 0.05,
            min_routing_entropy_noise: 0.05,
            max_alpha_residual: 0.1,
        }
    }
}
