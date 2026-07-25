use annp_core::MicroBlockConfig;
use annp_model::{ANNPModel, RoutingTable};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCheckpoint {
    pub node_id: usize,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub alpha: f32,
    pub cumulative_sequence_len: u64,
    pub activation_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub stage_completed: usize,
    pub epoch_completed: usize,
    pub config: MicroBlockConfig,
    pub nodes: Vec<NodeCheckpoint>,
    pub routing_tables: Vec<RoutingTable>,
}

impl ModelCheckpoint {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, content)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let ckpt: Self = serde_json::from_str(&content)?;
        Ok(ckpt)
    }

    pub fn apply_to_model(&self, model: &mut ANNPModel) {
        for node_ckpt in &self.nodes {
            if node_ckpt.node_id < model.nodes.len() {
                let node = &mut model.nodes[node_ckpt.node_id];
                node.w_gate = node_ckpt.w_gate.clone();
                node.w_up = node_ckpt.w_up.clone();
                node.w_down = node_ckpt.w_down.clone();
                node.alpha = node_ckpt.alpha;
                node.cumulative_sequence_len = node_ckpt.cumulative_sequence_len;
                node.activation_count = node_ckpt.activation_count;
            }
        }
        model.topology.routing_tables = self.routing_tables.clone();
    }

    pub fn extract_from_model(
        model: &ANNPModel,
        stage_completed: usize,
        epoch_completed: usize,
    ) -> Self {
        let nodes = model
            .nodes
            .iter()
            .map(|n| NodeCheckpoint {
                node_id: n.node_id,
                w_gate: n.w_gate.clone(),
                w_up: n.w_up.clone(),
                w_down: n.w_down.clone(),
                alpha: n.alpha,
                cumulative_sequence_len: n.cumulative_sequence_len,
                activation_count: n.activation_count,
            })
            .collect();

        Self {
            stage_completed,
            epoch_completed,
            config: model.config.clone(),
            nodes,
            routing_tables: model.topology.routing_tables.clone(),
        }
    }
}
