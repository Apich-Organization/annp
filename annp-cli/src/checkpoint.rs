use annp_core::{MicroBlockConfig, OnlineStats};
use annp_model::{ANNPModel, RoutingTable, Subnode};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const ANNPB_MAGIC: &[u8; 4] = b"ANNP";
// ANNPB versioning strategy: forward-compatible. Older checkpoints can be loaded by
// newer code. Every time a new persisted field is added, this version MUST be incremented.
const ANNPB_VERSION: u32 = 9;

/// Per-subnode checkpoint data.
///
/// `credit_stats` is saved and restored (NOT reset on checkpoint load).
/// It is the statistical foundation for Thompson Sampling: after resume,
/// subnode competition resumes immediately with meaningful credit histories
/// rather than all subnodes starting equal from count=0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnodeCheckpoint {
    pub subnode_id: usize,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub alpha: f32,
    pub activation_count: u64,
    pub credit_stats: OnlineStats,
    pub health: f32,
}

/// Per-node checkpoint data.
///
/// `last_p_in`, `last_prediction`, and `last_token_id` are saved to preserve
/// TD learning state across checkpoints. Without these, the first sequence after
/// resume would have no valid `last_token_id`, skipping one TD update step.
///
/// These fields are the inter-sequence "cold state": they are cleared by
/// `reset_state()` at sequence boundaries during training, but the checkpoint
/// captures them between sequences so resume starts from a valid TD context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCheckpoint {
    pub node_id: usize,
    pub split_count: u32,
    pub cumulative_sequence_len: u64,
    pub activation_count: u64,
    pub subnodes: Vec<SubnodeCheckpoint>,
    pub fast_weight: Vec<f32>,
    pub cumulative_energy: f32,
    pub last_p_in: Vec<f32>,
    pub last_prediction: Vec<f32>,
    pub last_token_id: Option<u32>,
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
    /// Save checkpoint to file.
    /// Uses `.annpb` binary format if the extension is `.annpb`, otherwise JSON.
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

    /// Load checkpoint from file.
    /// Auto-detects `.annpb` binary format by magic header or file extension; falls back to JSON.
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

    /// Zero-overhead binary serializer (.annpb v3 with Subnode hierarchy)
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

        // Write Config payload as UTF-8 JSON String
        let config_json = serde_json::to_string(&self.config)?;
        let config_bytes = config_json.as_bytes();
        file.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
        file.write_all(config_bytes)?;

        file.write_all(&num_nodes_u32.to_le_bytes())?;
        file.write_all(&num_routes_u32.to_le_bytes())?;

        // 1. Write Node Micro-Block & Subnode Weights
        for node in &self.nodes {
            file.write_all(&(node.node_id as u32).to_le_bytes())?;
            file.write_all(&node.split_count.to_le_bytes())?;
            file.write_all(&node.cumulative_sequence_len.to_le_bytes())?;
            file.write_all(&node.activation_count.to_le_bytes())?;

            let write_f32_vec = |f: &mut File,
                                 v: &Vec<f32>|
             -> Result<(), Box<dyn std::error::Error>> {
                f.write_all(&(v.len() as u32).to_le_bytes())?;
                if !v.is_empty() {
                    let b =
                        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
                    f.write_all(b)?;
                }
                Ok(())
            };
            write_f32_vec(&mut file, &node.fast_weight)?;
            file.write_all(&node.cumulative_energy.to_le_bytes())?;
            write_f32_vec(&mut file, &node.last_p_in)?;
            write_f32_vec(&mut file, &node.last_prediction)?;
            file.write_all(&node.last_token_id.unwrap_or(u32::MAX).to_le_bytes())?;

            file.write_all(&(node.subnodes.len() as u32).to_le_bytes())?;

            for sub in &node.subnodes {
                file.write_all(&(sub.subnode_id as u32).to_le_bytes())?;
                file.write_all(&sub.alpha.to_le_bytes())?;
                file.write_all(&sub.activation_count.to_le_bytes())?;

                file.write_all(&sub.credit_stats.count.to_le_bytes())?;
                file.write_all(&sub.credit_stats.mean.to_le_bytes())?;
                file.write_all(&sub.credit_stats.m2.to_le_bytes())?;
                file.write_all(&sub.credit_stats.gamma.to_le_bytes())?;
                file.write_all(&sub.health.to_le_bytes())?;

                let gate_bytes = unsafe {
                    std::slice::from_raw_parts(
                        sub.w_gate.as_ptr() as *const u8,
                        sub.w_gate.len() * 4,
                    )
                };
                let up_bytes = unsafe {
                    std::slice::from_raw_parts(sub.w_up.as_ptr() as *const u8, sub.w_up.len() * 4)
                };
                let down_bytes = unsafe {
                    std::slice::from_raw_parts(
                        sub.w_down.as_ptr() as *const u8,
                        sub.w_down.len() * 4,
                    )
                };

                file.write_all(&(sub.w_gate.len() as u32).to_le_bytes())?;
                file.write_all(gate_bytes)?;
                file.write_all(&(sub.w_up.len() as u32).to_le_bytes())?;
                file.write_all(up_bytes)?;
                file.write_all(&(sub.w_down.len() as u32).to_le_bytes())?;
                file.write_all(down_bytes)?;
            }
        }

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

            file.write_all(&(rt.edge_credit.len() as u32).to_le_bytes())?;
            for stats in &rt.edge_credit {
                file.write_all(&stats.count.to_le_bytes())?;
                file.write_all(&stats.mean.to_le_bytes())?;
                file.write_all(&stats.m2.to_le_bytes())?;
                file.write_all(&stats.gamma.to_le_bytes())?;
            }
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
        let version = u32::from_le_bytes(buf4);

        file.read_exact(&mut buf4)?;
        let stage_completed = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let epoch_completed = u32::from_le_bytes(buf4) as usize;

        let config = if version >= 2 {
            file.read_exact(&mut buf4)?;
            let config_len = u32::from_le_bytes(buf4) as usize;
            let mut config_buf = vec![0u8; config_len];
            file.read_exact(&mut config_buf)?;
            serde_json::from_slice(&config_buf)?
        } else {
            MicroBlockConfig::default()
        };

        file.read_exact(&mut buf4)?;
        let num_nodes = u32::from_le_bytes(buf4) as usize;
        file.read_exact(&mut buf4)?;
        let num_routes = u32::from_le_bytes(buf4) as usize;

        let mut nodes = Vec::with_capacity(num_nodes);

        if version >= 3 {
            // v3 Subnode format
            for _ in 0..num_nodes {
                file.read_exact(&mut buf4)?;
                let node_id = u32::from_le_bytes(buf4) as usize;
                file.read_exact(&mut buf4)?;
                let split_count = u32::from_le_bytes(buf4);
                file.read_exact(&mut buf8)?;
                let cumulative_sequence_len = u64::from_le_bytes(buf8);
                file.read_exact(&mut buf8)?;
                let activation_count = u64::from_le_bytes(buf8);

                let mut fast_weight = Vec::new();
                let mut cumulative_energy = 0.0f32;
                let mut last_p_in = Vec::new();
                let mut last_prediction = Vec::new();
                let mut last_token_id = None;

                if version >= 5 {
                    let read_f32_vec =
                        |file: &mut File,
                         buf4: &mut [u8; 4]|
                         -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                            file.read_exact(buf4)?;
                            let len = u32::from_le_bytes(*buf4) as usize;
                            let mut vec = vec![0.0f32; len];
                            if len > 0 {
                                let bytes = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        vec.as_mut_ptr() as *mut u8,
                                        len * 4,
                                    )
                                };
                                file.read_exact(bytes)?;
                            }
                            Ok(vec)
                        };
                    fast_weight = read_f32_vec(&mut file, &mut buf4)?;
                    file.read_exact(&mut buf4)?;
                    cumulative_energy = f32::from_le_bytes(buf4);

                    last_p_in = read_f32_vec(&mut file, &mut buf4)?;
                    last_prediction = read_f32_vec(&mut file, &mut buf4)?;

                    file.read_exact(&mut buf4)?;
                    let l_tid = u32::from_le_bytes(buf4);
                    if l_tid != u32::MAX {
                        last_token_id = Some(l_tid);
                    }
                }

                file.read_exact(&mut buf4)?;
                let num_subnodes = u32::from_le_bytes(buf4) as usize;
                let mut subnodes = Vec::with_capacity(num_subnodes);

                for _ in 0..num_subnodes {
                    file.read_exact(&mut buf4)?;
                    let subnode_id = u32::from_le_bytes(buf4) as usize;
                    file.read_exact(&mut buf4)?;
                    let alpha = f32::from_le_bytes(buf4);
                    file.read_exact(&mut buf8)?;
                    let sub_act_count = u64::from_le_bytes(buf8);

                    let mut credit_stats = OnlineStats::default();
                    if version >= 4 {
                        if version >= 7 {
                            file.read_exact(&mut buf4)?;
                            credit_stats.count = f32::from_le_bytes(buf4);
                        } else {
                            file.read_exact(&mut buf8)?;
                            credit_stats.count = u64::from_le_bytes(buf8) as f32;
                        }
                        file.read_exact(&mut buf4)?;
                        credit_stats.mean = f32::from_le_bytes(buf4);
                        file.read_exact(&mut buf4)?;
                        credit_stats.m2 = f32::from_le_bytes(buf4);
                        if version >= 9 {
                            file.read_exact(&mut buf4)?;
                            credit_stats.gamma = f32::from_le_bytes(buf4);
                        }
                    }

                    let mut health = 1.0f32;
                    if version >= 8 {
                        file.read_exact(&mut buf4)?;
                        health = f32::from_le_bytes(buf4);
                    }

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
                    let up_bytes = unsafe {
                        std::slice::from_raw_parts_mut(w_up.as_mut_ptr() as *mut u8, up_len * 4)
                    };
                    file.read_exact(up_bytes)?;

                    // Read W_down
                    file.read_exact(&mut buf4)?;
                    let down_len = u32::from_le_bytes(buf4) as usize;
                    let mut w_down = vec![0.0f32; down_len];
                    let down_bytes = unsafe {
                        std::slice::from_raw_parts_mut(w_down.as_mut_ptr() as *mut u8, down_len * 4)
                    };
                    file.read_exact(down_bytes)?;

                    subnodes.push(SubnodeCheckpoint {
                        subnode_id,
                        w_gate,
                        w_up,
                        w_down,
                        alpha,
                        activation_count: sub_act_count,
                        credit_stats,
                        health,
                    });
                }

                nodes.push(NodeCheckpoint {
                    node_id,
                    split_count,
                    cumulative_sequence_len,
                    activation_count,
                    subnodes,
                    fast_weight,
                    cumulative_energy,
                    last_p_in,
                    last_prediction,
                    last_token_id,
                });
            }
        } else {
            // Backward compatibility for v1/v2 legacy single-subnode format
            for _ in 0..num_nodes {
                file.read_exact(&mut buf4)?;
                let node_id = u32::from_le_bytes(buf4) as usize;

                file.read_exact(&mut buf4)?;
                let alpha = f32::from_le_bytes(buf4);

                file.read_exact(&mut buf8)?;
                let cumulative_sequence_len = u64::from_le_bytes(buf8);

                let activation_count = if version >= 2 {
                    file.read_exact(&mut buf8)?;
                    u64::from_le_bytes(buf8)
                } else {
                    0
                };

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
                let up_bytes = unsafe {
                    std::slice::from_raw_parts_mut(w_up.as_mut_ptr() as *mut u8, up_len * 4)
                };
                file.read_exact(up_bytes)?;

                // Read W_down
                file.read_exact(&mut buf4)?;
                let down_len = u32::from_le_bytes(buf4) as usize;
                let mut w_down = vec![0.0f32; down_len];
                let down_bytes = unsafe {
                    std::slice::from_raw_parts_mut(w_down.as_mut_ptr() as *mut u8, down_len * 4)
                };
                file.read_exact(down_bytes)?;

                let primary_subnode = SubnodeCheckpoint {
                    subnode_id: 0,
                    w_gate,
                    w_up,
                    w_down,
                    alpha,
                    activation_count,
                    credit_stats: OnlineStats::default(),
                    health: 1.0,
                };

                nodes.push(NodeCheckpoint {
                    node_id,
                    split_count: 0,
                    cumulative_sequence_len,
                    activation_count,
                    subnodes: vec![primary_subnode],
                    fast_weight: Vec::new(),
                    cumulative_energy: 0.0,
                    last_p_in: Vec::new(),
                    last_prediction: Vec::new(),
                    last_token_id: None,
                });
            }
        }

        // 3. Read P2P Grid Q-Routing Tables
        let mut routing_tables = Vec::with_capacity(num_routes);
        for _ in 0..num_routes {
            file.read_exact(&mut buf4)?;
            let d_head = u32::from_le_bytes(buf4) as usize;

            file.read_exact(&mut buf4)?;
            let neighbors_len = u32::from_le_bytes(buf4) as usize;
            let mut neighbors = Vec::with_capacity(neighbors_len);
            for _ in 0..neighbors_len {
                file.read_exact(&mut buf4)?;
                neighbors.push(u32::from_le_bytes(buf4) as usize);
            }

            file.read_exact(&mut buf4)?;
            let q_weight_len = u32::from_le_bytes(buf4) as usize;
            let mut weights = vec![0.0f32; q_weight_len];
            let q_weight_bytes = unsafe {
                std::slice::from_raw_parts_mut(weights.as_mut_ptr() as *mut u8, q_weight_len * 4)
            };
            file.read_exact(q_weight_bytes)?;

            let mut edge_credit = Vec::new();
            if version >= 5 {
                file.read_exact(&mut buf4)?;
                let ec_len = u32::from_le_bytes(buf4) as usize;
                for _ in 0..ec_len {
                    let count_f32 = if version >= 7 {
                        file.read_exact(&mut buf4)?;
                        f32::from_le_bytes(buf4)
                    } else {
                        file.read_exact(&mut buf8)?;
                        u64::from_le_bytes(buf8) as f32
                    };
                    file.read_exact(&mut buf4)?;
                    let mean = f32::from_le_bytes(buf4);
                    file.read_exact(&mut buf4)?;
                    let m2 = f32::from_le_bytes(buf4);
                    let mut gamma = 0.99;
                    if version >= 9 {
                        file.read_exact(&mut buf4)?;
                        gamma = f32::from_le_bytes(buf4);
                    }
                    edge_credit.push(OnlineStats {
                        count: count_f32,
                        mean,
                        m2,
                        gamma,
                    });
                }
            }

            routing_tables.push(RoutingTable {
                d_head,
                neighbors,
                weights,
                edge_credit,
            });
        }

        Ok(Self {
            stage_completed,
            epoch_completed,
            config,
            nodes,
            routing_tables,
        })
    }

    /// Restore checkpoint state into a live model instance.
    pub fn apply_to_model(&self, model: &mut ANNPModel) {
        model.config = self.config.clone();

        for node_ckpt in &self.nodes {
            if node_ckpt.node_id < model.nodes.len() {
                let node = &mut model.nodes[node_ckpt.node_id];
                node.split_count = node_ckpt.split_count;
                node.cumulative_sequence_len = node_ckpt.cumulative_sequence_len;
                node.activation_count = node_ckpt.activation_count;

                node.fast_weight.clone_from(&node_ckpt.fast_weight);
                node.cumulative_energy = node_ckpt.cumulative_energy;
                node.last_p_in.clone_from(&node_ckpt.last_p_in);
                node.last_prediction.clone_from(&node_ckpt.last_prediction);
                node.last_token_id = node_ckpt.last_token_id;

                let use_cuda = node.use_cuda;
                node.subnodes = node_ckpt
                    .subnodes
                    .iter()
                    .map(|s| {
                        let d_weights = if use_cuda {
                            Some(annp_cuda::ffi::CudaDeviceWeights::new(
                                &s.w_gate, &s.w_up, &s.w_down,
                            ))
                        } else {
                            None
                        };
                        Subnode {
                            subnode_id: s.subnode_id,
                            w_gate: s.w_gate.clone(),
                            w_up: s.w_up.clone(),
                            w_down: s.w_down.clone(),
                            alpha: s.alpha,
                            activation_count: s.activation_count,
                            credit_stats: s.credit_stats,
                            health: s.health,
                            d_weights,
                        }
                    })
                    .collect();
            }
        }

        if !self.routing_tables.is_empty() {
            model.topology.routing_tables = self.routing_tables.clone();
            model.topology.num_nodes = model.nodes.len();
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
                split_count: n.split_count,
                cumulative_sequence_len: n.cumulative_sequence_len,
                activation_count: n.activation_count,
                subnodes: n
                    .subnodes
                    .iter()
                    .map(|s| SubnodeCheckpoint {
                        subnode_id: s.subnode_id,
                        w_gate: s.w_gate.clone(),
                        w_up: s.w_up.clone(),
                        w_down: s.w_down.clone(),
                        alpha: s.alpha,
                        activation_count: s.activation_count,
                        credit_stats: s.credit_stats,
                        health: s.health,
                    })
                    .collect(),
                fast_weight: n.fast_weight.clone(),
                cumulative_energy: n.cumulative_energy,
                last_p_in: n.last_p_in.clone(),
                last_prediction: n.last_prediction.clone(),
                last_token_id: n.last_token_id,
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_checkpoint_binary_and_json_roundtrip() {
        let config = MicroBlockConfig {
            mesh_rows: 4,
            mesh_cols: 4,
            d_head: 32,
            ..MicroBlockConfig::default()
        };

        let model = ANNPModel::new(16, 2, config.clone(), Device::Cpu);
        let mut ckpt = ModelCheckpoint::extract_from_model(&model, 1, 5);
        ckpt.nodes[0].activation_count = 123;

        let tmp_dir = std::env::temp_dir();
        let bin_path = tmp_dir.join("test_checkpoint_tmp.annpb");
        let json_path = tmp_dir.join("test_checkpoint_tmp.json");

        // 1. Binary (.annpb) roundtrip
        ckpt.save(&bin_path).unwrap();

        let loaded_bin = ModelCheckpoint::load(&bin_path).unwrap();
        assert_eq!(loaded_bin.stage_completed, 1);
        assert_eq!(loaded_bin.epoch_completed, 5);
        assert_eq!(loaded_bin.config.d_head, 32);
        assert_eq!(loaded_bin.nodes[0].activation_count, 123);
        assert_eq!(loaded_bin.nodes[0].subnodes.len(), 1);

        // 2. JSON roundtrip
        ckpt.save(&json_path).unwrap();

        let loaded_json = ModelCheckpoint::load(&json_path).unwrap();
        assert_eq!(loaded_json.stage_completed, 1);
        assert_eq!(loaded_json.config.d_head, 32);
        assert_eq!(loaded_json.nodes[0].activation_count, 123);
        assert_eq!(loaded_json.nodes[0].subnodes.len(), 1);

        // 3. Apply to new model
        let mut target_model = ANNPModel::new(16, 2, MicroBlockConfig::default(), Device::Cpu);
        loaded_bin.apply_to_model(&mut target_model);
        assert_eq!(target_model.nodes.len(), 16);
        assert_eq!(target_model.config.d_head, 32);
        assert_eq!(target_model.nodes[0].activation_count, 123);

        let _ = fs::remove_file(bin_path);
        let _ = fs::remove_file(json_path);
    }
}
