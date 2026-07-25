use annp_core::MicroBlockConfig;
use annp_model::{ANNPModel, RoutingTable};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const ANNPB_MAGIC: &[u8; 4] = b"ANNP";
const ANNPB_VERSION: u32 = 1;

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
    pub w_egress: Vec<f32>,
}

impl ModelCheckpoint {
    /// Save checkpoint to file (.annpb binary format if extension is .annpb or default, otherwise JSON)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }

        let is_binary = p.extension().and_then(|ext| ext.to_str()) == Some("annpb");

        if is_binary {
            self.save_binary(p)?;
        } else {
            let content = serde_json::to_string_pretty(self)?;
            fs::write(p, content)?;
        }
        Ok(())
    }

    /// Load checkpoint from file (auto-detects .annpb binary format by magic header or extension)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let p = path.as_ref();
        let mut file = File::open(p)?;
        let mut magic = [0u8; 4];

        if file.read_exact(&mut magic).is_ok() && &magic == ANNPB_MAGIC {
            // Binary ANNPB format
            Self::load_binary(p)
        } else {
            // Text JSON format
            let content = fs::read_to_string(p)?;
            let ckpt: Self = serde_json::from_str(&content)?;
            Ok(ckpt)
        }
    }

    /// Zero-overhead binary serializer (.annpb)
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        file.write_all(ANNPB_MAGIC)?;
        file.write_all(&ANNPB_VERSION.to_le_bytes())?;

        let stage_u32 = self.stage_completed as u32;
        let epoch_u32 = self.epoch_completed as u32;
        let num_nodes_u32 = self.nodes.len() as u32;
        let num_routes_u32 = self.routing_tables.len() as u32;

        file.write_all(&stage_u32.to_le_bytes())?;
        file.write_all(&epoch_u32.to_le_bytes())?;
        file.write_all(&num_nodes_u32.to_le_bytes())?;
        file.write_all(&num_routes_u32.to_le_bytes())?;

        // 1. Write Node Micro-Block & Non-linear FFN Weights
        for node in &self.nodes {
            file.write_all(&(node.node_id as u32).to_le_bytes())?;
            file.write_all(&node.alpha.to_le_bytes())?;
            file.write_all(&node.cumulative_sequence_len.to_le_bytes())?;

            let gate_bytes = unsafe {
                std::slice::from_raw_parts(node.w_gate.as_ptr() as *const u8, node.w_gate.len() * 4)
            };
            let up_bytes = unsafe {
                std::slice::from_raw_parts(node.w_up.as_ptr() as *const u8, node.w_up.len() * 4)
            };
            let down_bytes = unsafe {
                std::slice::from_raw_parts(node.w_down.as_ptr() as *const u8, node.w_down.len() * 4)
            };

            file.write_all(&(node.w_gate.len() as u32).to_le_bytes())?;
            file.write_all(gate_bytes)?;
            file.write_all(&(node.w_up.len() as u32).to_le_bytes())?;
            file.write_all(up_bytes)?;
            file.write_all(&(node.w_down.len() as u32).to_le_bytes())?;
            file.write_all(down_bytes)?;
        }

        // 2. Write Egress Serializer Weights
        let egress_bytes = unsafe {
            std::slice::from_raw_parts(self.w_egress.as_ptr() as *const u8, self.w_egress.len() * 4)
        };
        file.write_all(&(self.w_egress.len() as u32).to_le_bytes())?;
        file.write_all(egress_bytes)?;

        // 3. Write P2P Grid Q-Routing Tables
        for rt in &self.routing_tables {
            file.write_all(&(rt.d_head as u32).to_le_bytes())?;
            file.write_all(&(rt.neighbors.len() as u32).to_le_bytes())?;
            for &n in &rt.neighbors {
                file.write_all(&(n as u32).to_le_bytes())?;
            }

            let q_weight_bytes = unsafe {
                std::slice::from_raw_parts(rt.weights.as_ptr() as *const u8, rt.weights.len() * 4)
            };
            file.write_all(&(rt.weights.len() as u32).to_le_bytes())?;
            file.write_all(q_weight_bytes)?;
        }

        Ok(())
    }

    /// Fast binary deserializer (.annpb)
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != ANNPB_MAGIC {
            return Err("Invalid ANNPB magic header".into());
        }

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        file.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);

        file.read_exact(&mut buf4)?;
        let stage_completed = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let epoch_completed = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let num_nodes = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let num_routes = u32::from_le_bytes(buf4) as usize;

        let mut nodes = Vec::with_capacity(num_nodes);

        // 1. Read Node Micro-Block Weights
        for _ in 0..num_nodes {
            file.read_exact(&mut buf4)?;
            let node_id = u32::from_le_bytes(buf4) as usize;

            file.read_exact(&mut buf4)?;
            let alpha = f32::from_le_bytes(buf4);

            file.read_exact(&mut buf8)?;
            let cumulative_sequence_len = u64::from_le_bytes(buf8);

            // Read W_gate
            file.read_exact(&mut buf4)?;
            let gate_len = u32::from_le_bytes(buf4) as usize;
            let mut w_gate = vec![0.0f32; gate_len];
            let gate_bytes = unsafe {
                std::slice::from_raw_parts_mut(w_gate.as_mut_ptr() as *mut u8, gate_len * 4)
            };
            file.read_exact(gate_bytes)?;

            // Read W_up
            file.read_exact(&mut buf4)?;
            let up_len = u32::from_le_bytes(buf4) as usize;
            let mut w_up = vec![0.0f32; up_len];
            let up_bytes =
                unsafe { std::slice::from_raw_parts_mut(w_up.as_mut_ptr() as *mut u8, up_len * 4) };
            file.read_exact(up_bytes)?;

            // Read W_down
            file.read_exact(&mut buf4)?;
            let down_len = u32::from_le_bytes(buf4) as usize;
            let mut w_down = vec![0.0f32; down_len];
            let down_bytes = unsafe {
                std::slice::from_raw_parts_mut(w_down.as_mut_ptr() as *mut u8, down_len * 4)
            };
            file.read_exact(down_bytes)?;

            nodes.push(NodeCheckpoint {
                node_id,
                w_gate,
                w_up,
                w_down,
                alpha,
                cumulative_sequence_len,
                activation_count: 0,
            });
        }

        // 2. Read Egress Serializer Weights
        file.read_exact(&mut buf4)?;
        let egress_len = u32::from_le_bytes(buf4) as usize;
        let mut w_egress = vec![0.0f32; egress_len];
        let egress_bytes = unsafe {
            std::slice::from_raw_parts_mut(w_egress.as_mut_ptr() as *mut u8, egress_len * 4)
        };
        file.read_exact(egress_bytes)?;

        // 3. Read Routing Tables
        let mut routing_tables = Vec::with_capacity(num_routes);
        for _ in 0..num_routes {
            file.read_exact(&mut buf4)?;
            let d_head = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let num_neighbors = u32::from_le_bytes(buf4) as usize;

            let mut neighbors = Vec::with_capacity(num_neighbors);
            for _ in 0..num_neighbors {
                file.read_exact(&mut buf4)?;
                neighbors.push(u32::from_le_bytes(buf4) as usize);
            }

            file.read_exact(&mut buf4)?;
            let weights_len = u32::from_le_bytes(buf4) as usize;
            let mut weights = vec![0.0f32; weights_len];
            let weight_bytes = unsafe {
                std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, weights_len * 4)
            };
            file.read_exact(weight_bytes)?;

            routing_tables.push(RoutingTable {
                d_head,
                neighbors,
                weights,
            });
        }

        let default_config = MicroBlockConfig::default();

        Ok(Self {
            stage_completed,
            epoch_completed,
            config: default_config,
            nodes,
            routing_tables,
            w_egress,
        })
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
        if !self.w_egress.is_empty() && self.w_egress.len() == model.serializer.w_egress.len() {
            model.serializer.w_egress = self.w_egress.clone();
        }
        if !self.routing_tables.is_empty() {
            model.topology.routing_tables = self.routing_tables.clone();
        }
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
            w_egress: model.serializer.w_egress.clone(),
        }
    }
}
