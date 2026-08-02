use crate::checkpoint::ModelCheckpoint;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
struct TopologyExportJson<'a> {
    stage_completed: usize,
    epoch_completed: usize,
    num_nodes: usize,
    mesh_rows: usize,
    mesh_cols: usize,
    d_head: usize,
    d_model: usize,
    routing_tables: &'a Vec<annp_model::RoutingTable>,
    node_summaries: Vec<NodeExportSummary>,
}

#[derive(Serialize)]
struct NodeExportSummary {
    node_id: usize,
    split_count: u32,
    subnode_count: usize,
    activation_count: u64,
    cumulative_energy: f32,
    mean_subnode_health: f32,
}

pub fn execute_export(
    checkpoint_path: PathBuf,
    topology_out: PathBuf,
    format_opt: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !checkpoint_path.exists() {
        return Err(format!("Checkpoint file {:?} does not exist", checkpoint_path).into());
    }

    println!("Loading Checkpoint from: {:?}", checkpoint_path);
    let ckpt = ModelCheckpoint::load(&checkpoint_path)?;

    // Determine export format: explicit flag or infer from output file extension
    let format = format_opt.unwrap_or_else(|| {
        topology_out
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("json")
            .to_lowercase()
    });

    if let Some(parent) = topology_out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    match format.as_str() {
        "dot" | "gv" => {
            export_dot(&ckpt, &topology_out)?;
        }
        "csv" => {
            export_csv(&ckpt, &topology_out)?;
        }
        "summary" | "txt" => {
            export_summary(&ckpt, &topology_out)?;
        }
        _ => {
            export_json(&ckpt, &topology_out)?;
        }
    }

    Ok(())
}

fn export_json(ckpt: &ModelCheckpoint, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let node_summaries: Vec<NodeExportSummary> = ckpt
        .nodes
        .iter()
        .map(|n| {
            let mean_health = if n.subnodes.is_empty() {
                0.0
            } else {
                n.subnodes.iter().map(|s| s.health).sum::<f32>() / (n.subnodes.len() as f32)
            };
            NodeExportSummary {
                node_id: n.node_id,
                split_count: n.split_count,
                subnode_count: n.subnodes.len(),
                activation_count: n.activation_count,
                cumulative_energy: n.cumulative_energy,
                mean_subnode_health: mean_health,
            }
        })
        .collect();

    let export_payload = TopologyExportJson {
        stage_completed: ckpt.stage_completed,
        epoch_completed: ckpt.epoch_completed,
        num_nodes: ckpt.nodes.len(),
        mesh_rows: ckpt.config.mesh_rows,
        mesh_cols: ckpt.config.mesh_cols,
        d_head: ckpt.config.d_head,
        d_model: ckpt.config.d_model(),
        routing_tables: &ckpt.routing_tables,
        node_summaries,
    };

    let content = serde_json::to_string_pretty(&export_payload)?;
    fs::write(out, content)?;
    println!(
        "Successfully exported JSON Topology ({} nodes, {} routing tables) to: {:?}",
        ckpt.nodes.len(),
        ckpt.routing_tables.len(),
        out
    );
    Ok(())
}

fn export_dot(ckpt: &ModelCheckpoint, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut dot = String::new();
    dot.push_str("digraph ANNP_Mesh_Topology {\n");
    dot.push_str("  graph [rankdir=LR, bgcolor=\"#ffffff\", fontname=\"Helvetica\"];\n");
    dot.push_str("  node [shape=box, style=\"rounded,filled\", fillcolor=\"#f0f4f8\", color=\"#627d98\", fontname=\"Helvetica\", fontsize=10];\n");
    dot.push_str("  edge [fontname=\"Helvetica\", fontsize=8, color=\"#829ab1\"];\n\n");

    // Node declarations
    for node in &ckpt.nodes {
        let subnode_count = node.subnodes.len();
        let label = format!(
            "Node {}\\nsubnodes: {}\\nenergy: {:.2}",
            node.node_id, subnode_count, node.cumulative_energy
        );
        dot.push_str(&format!("  node_{} [label=\"{}\"];\n", node.node_id, label));
    }
    dot.push('\n');

    // Edge declarations from routing tables
    for (src_id, rt) in ckpt.routing_tables.iter().enumerate() {
        for (edge_idx, &dst_id) in rt.neighbors.iter().enumerate() {
            let credit_str = if edge_idx < rt.edge_credit.len() {
                let stats = &rt.edge_credit[edge_idx];
                if stats.count > 0.0 {
                    format!("μ={:.3}, n={:.0}", stats.mean, stats.count)
                } else {
                    "init".to_string()
                }
            } else {
                "init".to_string()
            };

            let penwidth = if edge_idx < rt.edge_credit.len() && rt.edge_credit[edge_idx].mean > 0.0
            {
                "1.8"
            } else {
                "0.8"
            };

            dot.push_str(&format!(
                "  node_{} -> node_{} [label=\"{}\", penwidth={}];\n",
                src_id, dst_id, credit_str, penwidth
            ));
        }
    }

    dot.push_str("}\n");
    fs::write(out, dot)?;
    println!("Successfully exported Graphviz DOT Topology to: {:?}", out);
    Ok(())
}

fn export_csv(ckpt: &ModelCheckpoint, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut csv = String::new();
    csv.push_str("source_node,target_node,edge_index,mean_credit,variance_credit,observation_count,weight_norm\n");

    let d_head = ckpt.config.d_head;
    for (src_id, rt) in ckpt.routing_tables.iter().enumerate() {
        let num_neighbors = rt.neighbors.len();
        for (edge_idx, &dst_id) in rt.neighbors.iter().enumerate() {
            let (mean, var, count) = if edge_idx < rt.edge_credit.len() {
                let stats = &rt.edge_credit[edge_idx];
                (stats.mean, stats.variance(), stats.count)
            } else {
                (0.0, 0.0, 0.0)
            };

            // Compute weight norm for this edge
            let mut sq_sum = 0.0f32;
            for d in 0..d_head {
                let idx = d * num_neighbors + edge_idx;
                if idx < rt.weights.len() {
                    let w = rt.weights[idx];
                    sq_sum += w * w;
                }
            }
            let weight_norm = sq_sum.sqrt();

            csv.push_str(&format!(
                "{},{},{},{:.8},{:.8},{:.0},{:.8}\n",
                src_id, dst_id, edge_idx, mean, var, count, weight_norm
            ));
        }
    }

    fs::write(out, csv)?;
    println!("Successfully exported P2P Edge Adjacency CSV to: {:?}", out);
    Ok(())
}

fn export_summary(ckpt: &ModelCheckpoint, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut summary = String::new();
    summary.push_str(
        "================================================================================\n",
    );
    summary.push_str(
        "                    ANNP MODEL CHECKPOINT TOPOLOGY REPORT                       \n",
    );
    summary.push_str(
        "================================================================================\n\n",
    );

    summary.push_str(&format!("Stage Completed:     {}\n", ckpt.stage_completed));
    summary.push_str(&format!("Epoch Completed:     {}\n", ckpt.epoch_completed));
    summary.push_str(&format!(
        "Mesh Dimensions:     {} x {} ({} total nodes)\n",
        ckpt.config.mesh_rows,
        ckpt.config.mesh_cols,
        ckpt.nodes.len()
    ));
    summary.push_str(&format!(
        "Particle Dimension:  d_head={}, d_model={} ({} shards)\n",
        ckpt.config.d_head,
        ckpt.config.d_model(),
        ckpt.config.num_shards
    ));
    summary.push_str(&format!(
        "Hop Bounds:          min_hop={}, max_hop={}\n",
        ckpt.config.min_hop, ckpt.config.max_hop
    ));
    summary.push_str(&format!(
        "Initial Energy:      {:.4}\n",
        ckpt.config.initial_energy
    ));
    summary.push_str(&format!(
        "Weight Decay:        {:.6}\n\n",
        ckpt.config.weight_decay
    ));

    // Subnode and node stats
    let total_subnodes: usize = ckpt.nodes.iter().map(|n| n.subnodes.len()).sum();
    let avg_subnodes = total_subnodes as f32 / ckpt.nodes.len().max(1) as f32;
    let total_activations: u64 = ckpt.nodes.iter().map(|n| n.activation_count).sum();
    summary.push_str("--- Node Subnode Statistics ---\n");
    summary.push_str(&format!("Total Subnodes:      {}\n", total_subnodes));
    summary.push_str(&format!("Avg Subnodes/Node:   {:.2}\n", avg_subnodes));
    summary.push_str(&format!("Total Activations:   {}\n\n", total_activations));

    // Top routing edges by mean credit
    summary.push_str("--- Top Routing Edges by Thompson Credit ---\n");
    let mut edge_list = Vec::new();
    for (src_id, rt) in ckpt.routing_tables.iter().enumerate() {
        for (edge_idx, &dst_id) in rt.neighbors.iter().enumerate() {
            if edge_idx < rt.edge_credit.len() {
                let stats = &rt.edge_credit[edge_idx];
                if stats.count > 0.0 {
                    edge_list.push((src_id, dst_id, stats.mean, stats.count, stats.variance()));
                }
            }
        }
    }
    edge_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    if edge_list.is_empty() {
        summary.push_str("No active edge credit statistics recorded yet.\n");
    } else {
        summary.push_str(
            format!(
                "{:<10} {:<10} {:<15} {:<12} {:<15}\n",
                "Source", "Target", "Mean Credit", "Count", "Variance"
            )
            .as_str(),
        );
        summary.push_str("-----------------------------------------------------------------\n");
        for (src, dst, mean, count, var) in edge_list.iter().take(20) {
            summary.push_str(&format!(
                "{:<10} {:<10} {:<15.6} {:<12.0} {:<15.6}\n",
                src, dst, mean, count, var
            ));
        }
    }

    fs::write(out, summary)?;
    println!(
        "Successfully exported Human-Readable Summary Report to: {:?}",
        out
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_core::MicroBlockConfig;
    use annp_model::ANNPModel;
    use candle_core::Device;

    fn create_dummy_checkpoint() -> ModelCheckpoint {
        let config = MicroBlockConfig {
            mesh_rows: 2,
            mesh_cols: 2,
            d_head: 16,
            ..MicroBlockConfig::default()
        };
        let model = ANNPModel::new(4, 2, config, Device::Cpu);
        let mut ckpt = ModelCheckpoint::extract_from_model(&model, 1, 2);
        if !ckpt.routing_tables.is_empty() && !ckpt.routing_tables[0].edge_credit.is_empty() {
            ckpt.routing_tables[0].edge_credit[0].observe(0.75);
            ckpt.routing_tables[0].edge_credit[0].observe(0.85);
        }
        ckpt
    }

    #[test]
    fn test_export_all_formats() {
        let tmp_dir = std::env::temp_dir();
        let ckpt_path = tmp_dir.join("test_export_ckpt.annpb");
        let json_path = tmp_dir.join("test_export.json");
        let dot_path = tmp_dir.join("test_export.dot");
        let csv_path = tmp_dir.join("test_export.csv");
        let sum_path = tmp_dir.join("test_export.txt");

        let ckpt = create_dummy_checkpoint();
        ckpt.save(&ckpt_path).unwrap();

        // 1. JSON export
        let res_json = execute_export(
            ckpt_path.clone(),
            json_path.clone(),
            Some("json".to_string()),
        );
        assert!(res_json.is_ok());
        assert!(json_path.exists());
        let json_str = fs::read_to_string(&json_path).unwrap();
        assert!(json_str.contains("\"num_nodes\": 4"));
        assert!(json_str.contains("\"stage_completed\": 1"));

        // 2. DOT export
        let res_dot = execute_export(ckpt_path.clone(), dot_path.clone(), Some("dot".to_string()));
        assert!(res_dot.is_ok());
        assert!(dot_path.exists());
        let dot_str = fs::read_to_string(&dot_path).unwrap();
        assert!(dot_str.contains("digraph ANNP_Mesh_Topology"));
        assert!(dot_str.contains("node_0"));

        // 3. CSV export
        let res_csv = execute_export(ckpt_path.clone(), csv_path.clone(), Some("csv".to_string()));
        assert!(res_csv.is_ok());
        assert!(csv_path.exists());
        let csv_str = fs::read_to_string(&csv_path).unwrap();
        assert!(csv_str.starts_with("source_node,target_node,edge_index,mean_credit,variance_credit,observation_count,weight_norm"));

        // 4. Summary export
        let res_sum = execute_export(
            ckpt_path.clone(),
            sum_path.clone(),
            Some("summary".to_string()),
        );
        assert!(res_sum.is_ok());
        assert!(sum_path.exists());
        let sum_str = fs::read_to_string(&sum_path).unwrap();
        assert!(sum_str.contains("ANNP MODEL CHECKPOINT TOPOLOGY REPORT"));
        assert!(sum_str.contains("Stage Completed:     1"));

        // 5. Inferred extension export
        let inferred_dot = tmp_dir.join("test_inferred.gv");
        let res_inferred = execute_export(ckpt_path.clone(), inferred_dot.clone(), None);
        assert!(res_inferred.is_ok());
        let inferred_str = fs::read_to_string(&inferred_dot).unwrap();
        assert!(inferred_str.contains("digraph ANNP_Mesh_Topology"));

        let _ = fs::remove_file(ckpt_path);
        let _ = fs::remove_file(json_path);
        let _ = fs::remove_file(dot_path);
        let _ = fs::remove_file(csv_path);
        let _ = fs::remove_file(sum_path);
        let _ = fs::remove_file(inferred_dot);
    }
}
