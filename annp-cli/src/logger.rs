use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

pub struct AnnpLogger {
    file_writer: Option<Mutex<File>>,
    csv_writer: Option<Mutex<File>>,
    log_file_path: Option<PathBuf>,
}

impl AnnpLogger {
    pub fn new(log_dir: &Path, prefix: &str, custom_file: Option<&Path>) -> Self {
        if let Err(e) = fs::create_dir_all(log_dir) {
            eprintln!(
                "Warning: Could not create log directory {:?}: {}",
                log_dir, e
            );
        }

        let file_path = if let Some(p) = custom_file {
            p.to_path_buf()
        } else {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            log_dir.join(format!("{}_{}.log", prefix, timestamp))
        };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|e| {
                eprintln!("Warning: Failed to open log file {:?}: {}", file_path, e);
                e
            })
            .ok();

        let csv_file_path = file_path.with_extension("csv");
        let mut csv_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_file_path)
            .map_err(|e| {
                eprintln!(
                    "Warning: Failed to open CSV log file {:?}: {}",
                    csv_file_path, e
                );
                e
            })
            .ok();

        if let Some(ref mut csv) = csv_file {
            // Write CSV header if the file is newly created or empty
            if let Ok(metadata) = csv.metadata()
                && metadata.len() == 0
            {
                let _ = writeln!(
                    csv,
                    "timestamp,epoch,batch,step_loss,ema_loss,early_halting_rate,gini,avg_hops,avg_energy,memory_density,active_subnodes,credit_volatility,temporal_affinity"
                );
            }
        }

        if file.is_some() {
            println!(
                "Logging detailed execution metrics to file: {:?}",
                file_path
            );
        }

        Self {
            file_writer: file.map(Mutex::new),
            csv_writer: csv_file.map(Mutex::new),
            log_file_path: Some(file_path),
        }
    }

    /// Specialized Logger for External Task Fidelity Evaluation
    pub fn new_eval(log_dir: &Path, prefix: &str, custom_file: Option<&Path>) -> Self {
        if let Err(e) = fs::create_dir_all(log_dir) {
            eprintln!(
                "Warning: Could not create log directory {:?}: {}",
                log_dir, e
            );
        }

        let file_path = if let Some(p) = custom_file {
            p.to_path_buf()
        } else {
            let timestamp = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            log_dir.join(format!("{}_{}.log", prefix, timestamp))
        };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .map_err(|e| {
                eprintln!(
                    "Warning: Failed to open eval log file {:?}: {}",
                    file_path, e
                );
                e
            })
            .ok();

        let csv_file_path = file_path.with_extension("csv");
        let mut csv_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&csv_file_path)
            .map_err(|e| {
                eprintln!(
                    "Warning: Failed to open eval CSV log file {:?}: {}",
                    csv_file_path, e
                );
                e
            })
            .ok();

        if let Some(ref mut csv) = csv_file
            && let Ok(metadata) = csv.metadata()
            && metadata.len() == 0
        {
            let _ = writeln!(
                csv,
                "timestamp,batch,next_token_mse,next_token_cossim,next_token_mae,nmse,psnr_db,direct_recon_mse,internal_td_loss,early_halting_rate,gini,avg_hops,avg_energy,memory_density,active_subnodes"
            );
        }

        if file.is_some() {
            println!(
                "Logging evaluation fidelity metrics to file: {:?}",
                file_path
            );
        }

        Self {
            file_writer: file.map(Mutex::new),
            csv_writer: csv_file.map(Mutex::new),
            log_file_path: Some(file_path),
        }
    }

    pub fn get_log_path(&self) -> Option<&PathBuf> {
        self.log_file_path.as_ref()
    }

    pub fn log(&self, tag: &str, msg: &str) {
        let timestamp = chrono_timestamp();
        let formatted = format!("[{}] {} | {}\n", timestamp, tag, msg);

        // Always print to console
        print!("{}", formatted);

        // Write to log file if available
        if let Some(ref writer_mutex) = self.file_writer
            && let Ok(mut file) = writer_mutex.lock()
        {
            let _ = file.write_all(formatted.as_bytes());
            let _ = file.flush();
        }
    }

    pub fn log_step(
        &self,
        epoch: usize,
        total_epochs: usize,
        batch_idx: usize,
        step_loss: f32,
        rolling_loss: f32,
        metrics: &annp_model::BatchMetrics,
    ) {
        let msg = format!(
            "[Epoch {:2}/{:2} | Batch {:4}] Loss: {:.4} (EMA: {:.4}) | Halt: {:.2}% | Gini: {:.4} | Hops: {:.4}",
            epoch,
            total_epochs,
            batch_idx,
            step_loss,
            rolling_loss,
            metrics.early_halting_rate * 100.0,
            metrics.utilization_gini,
            metrics.avg_hop_count
        );
        self.log("TRAIN_STEP", &msg);

        if let Some(ref csv_mutex) = self.csv_writer
            && let Ok(mut csv) = csv_mutex.lock()
        {
            let timestamp = chrono_timestamp();
            let _ = writeln!(
                csv,
                "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                timestamp,
                epoch,
                batch_idx,
                step_loss,
                rolling_loss,
                metrics.early_halting_rate,
                metrics.utilization_gini,
                metrics.avg_hop_count,
                metrics.avg_signal_energy,
                metrics.avg_memory_density,
                metrics.avg_subnodes,
                metrics.avg_credit_volatility,
                metrics.avg_temporal_affinity
            );
        }
    }

    pub fn log_eval_step(
        &self,
        batch_idx: usize,
        total_batches: usize,
        eval_metrics: &EvalBatchMetrics,
        model_metrics: &annp_model::BatchMetrics,
    ) {
        let msg = format!(
            "[Eval Batch {:4}/{:4}] NextMSE: {:.6} | CosSim: {:.4} | PSNR: {:.2} dB | TD-Loss: {:.4} | Halt: {:.2}% | Hops: {:.2}",
            batch_idx,
            total_batches,
            eval_metrics.next_token_mse,
            eval_metrics.next_token_cossim,
            eval_metrics.psnr_db,
            eval_metrics.internal_td_loss,
            model_metrics.early_halting_rate * 100.0,
            model_metrics.avg_hop_count
        );
        self.log("EVAL_STEP", &msg);

        if let Some(ref csv_mutex) = self.csv_writer
            && let Ok(mut csv) = csv_mutex.lock()
        {
            let timestamp = chrono_timestamp();
            let _ = writeln!(
                csv,
                "{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                timestamp,
                batch_idx,
                eval_metrics.next_token_mse,
                eval_metrics.next_token_cossim,
                eval_metrics.next_token_mae,
                eval_metrics.nmse,
                eval_metrics.psnr_db,
                eval_metrics.direct_recon_mse,
                eval_metrics.internal_td_loss,
                model_metrics.early_halting_rate,
                model_metrics.utilization_gini,
                model_metrics.avg_hop_count,
                model_metrics.avg_signal_energy,
                model_metrics.avg_memory_density,
                model_metrics.avg_subnodes
            );
        }
    }

    pub fn log_epoch_summary(
        &self,
        epoch: usize,
        total_epochs: usize,
        avg_loss: f32,
        metrics: &annp_model::BatchMetrics,
    ) {
        let msg = format!(
            "Epoch {}/{} Metrics Summary:\n  - Avg Loss: {:.4}\n  - Avg Hop Count: {:.4}\n  - Early Halting Rate: {:.2}%\n  - Signal Energy (var): {:.4}\n  - Node Util Gini: {:.4}\n  - Attention Entropy: {:.4}\n  - Avg Active Subnodes: {:.4}\n  - Credit Volatility: {:.4}\n  - Mean Temporal Affinity: {:.4}",
            epoch,
            total_epochs,
            avg_loss,
            metrics.avg_hop_count,
            metrics.early_halting_rate * 100.0,
            metrics.avg_signal_energy,
            metrics.utilization_gini,
            metrics.avg_memory_density,
            metrics.avg_subnodes,
            metrics.avg_credit_volatility,
            metrics.avg_temporal_affinity
        );
        self.log("EPOCH_SUMMARY", &msg);
    }

    pub fn log_eval_summary(&self, summary: &EvalSummaryReport) {
        let recall_str = if let Some(ref r) = summary.associative_recall {
            format!(
                "\n  - Associative Key MSE: {:.8}\n  - Associative Key CosSim: {:.6}\n  - Key Retrieval Accuracy (CosSim > 0.9): {:.2}%",
                r.key_recall_mse,
                r.key_recall_cossim,
                r.retrieval_accuracy * 100.0
            )
        } else {
            String::new()
        };

        let msg = format!(
            "=================================================================================\n\
             ANNP External Task Fidelity Evaluation Summary ({}/{} Batches Evaluated):\n\
             ---------------------------------------------------------------------------------\n\
               External Task Accuracy Metrics:\n\
                 - Next-Token MSE (Mean +/- Std): {:.8} +/- {:.8}\n\
                 - Next-Token MSE (Median [Min, Max]): {:.8} [{:.8}, {:.8}]\n\
                 - Next-Token 95th Percentile MSE: {:.8}\n\
                 - Next-Token Cosine Similarity: {:.6} +/- {:.6}\n\
                 - Next-Token MAE: {:.8}\n\
                 - Normalized MSE (NMSE): {:.8}\n\
                 - Peak Signal-to-Noise Ratio (PSNR): {:.4} dB\n\
                 - Direct Autoencoding MSE: {:.8}\n\
               Dual-Metric Physical Dynamics:\n\
                 - Internal Local TD-Loss (Mean +/- Std): {:.8} +/- {:.8}\n\
                 - TD-Loss to External-MSE Ratio: {:.4}\n\
                 - Early Halting Rate: {:.2}%\n\
                 - Average Hop Count: {:.4}\n\
                 - Active Subnodes: {:.4}{}\n\
             =================================================================================",
            summary.total_batches,
            summary.total_batches,
            summary.next_token_mse_mean,
            summary.next_token_mse_std,
            summary.next_token_mse_median,
            summary.next_token_mse_min,
            summary.next_token_mse_max,
            summary.next_token_mse_p95,
            summary.next_token_cossim_mean,
            summary.next_token_cossim_std,
            summary.next_token_mae_mean,
            summary.nmse,
            summary.psnr_db,
            summary.direct_recon_mse_mean,
            summary.internal_td_loss_mean,
            summary.internal_td_loss_std,
            summary.td_to_mse_ratio,
            summary.early_halting_rate * 100.0,
            summary.avg_hops,
            summary.active_subnodes,
            recall_str
        );
        self.log("EVAL_SUMMARY", &msg);
    }

    pub fn log_hardening(
        &self,
        epoch: usize,
        links_before: usize,
        links_pruned: usize,
        nodes_grown: usize,
        spawn_details: &[(usize, usize, usize)], // (parent_a, parent_b, new_id)
    ) {
        let remaining_links = links_before.saturating_sub(links_pruned);
        let mut msg = format!(
            "Hardening Pass (Epoch {}): Pruned {} links (Remaining: {}). Spawned {} local subnodes.",
            epoch, links_pruned, remaining_links, nodes_grown
        );

        for (p_a, p_b, new_id) in spawn_details {
            msg.push_str(&format!(
                "\n  -> Node {}: subnodes {} -> {}",
                p_a, p_b, new_id
            ));
        }

        self.log("HARDENING", &msg);
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EvalBatchMetrics {
    pub next_token_mse: f32,
    pub next_token_cossim: f32,
    pub next_token_mae: f32,
    pub nmse: f32,
    pub psnr_db: f32,
    pub direct_recon_mse: f32,
    pub internal_td_loss: f32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AssociativeRecallReport {
    pub key_recall_mse: f32,
    pub key_recall_cossim: f32,
    pub retrieval_accuracy: f32,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EvalSummaryReport {
    pub checkpoint_path: Option<String>,
    pub dataset_name: String,
    pub total_batches: usize,
    pub next_token_mse_mean: f32,
    pub next_token_mse_std: f32,
    pub next_token_mse_median: f32,
    pub next_token_mse_min: f32,
    pub next_token_mse_max: f32,
    pub next_token_mse_p95: f32,
    pub next_token_cossim_mean: f32,
    pub next_token_cossim_std: f32,
    pub next_token_mae_mean: f32,
    pub nmse: f32,
    pub psnr_db: f32,
    pub direct_recon_mse_mean: f32,
    pub internal_td_loss_mean: f32,
    pub internal_td_loss_std: f32,
    pub td_to_mse_ratio: f32,
    pub early_halting_rate: f32,
    pub avg_hops: f32,
    pub active_subnodes: f32,
    pub associative_recall: Option<AssociativeRecallReport>,
}

fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now();
    if let Ok(dur) = now.duration_since(SystemTime::UNIX_EPOCH) {
        let secs = dur.as_secs();
        let millis = dur.subsec_millis();
        let hours = (secs / 3600) % 24;
        let mins = (secs / 60) % 60;
        let s = secs % 60;
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, s, millis)
    } else {
        "00:00:00.000".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_model::BatchMetrics;

    #[test]
    fn test_logger_creation_and_csv_precision() {
        let tmp_dir = std::env::temp_dir();
        let log_file = tmp_dir.join("test_annp_logger_run.log");
        let csv_file = tmp_dir.join("test_annp_logger_run.csv");

        let _ = fs::remove_file(&log_file);
        let _ = fs::remove_file(&csv_file);

        let logger = AnnpLogger::new(&tmp_dir, "test", Some(&log_file));
        logger.log("TEST_TAG", "Sample test log entry");

        let metrics = BatchMetrics {
            avg_signal_energy: 1.25,
            utilization_gini: 0.15,
            early_halting_rate: 0.05,
            avg_hop_count: 3.5,
            avg_memory_density: 0.85,
            avg_subnodes: 1.5,
            avg_credit_volatility: 0.02,
            avg_temporal_affinity: 0.95,
        };

        logger.log_step(1, 5, 42, 2.75, 2.8, &metrics);

        // Verify log file content
        assert!(log_file.exists());
        let log_text = fs::read_to_string(&log_file).unwrap();
        assert!(log_text.contains("Sample test log entry"));
        assert!(log_text.contains("TRAIN_STEP"));

        // Verify CSV file content and precision
        assert!(csv_file.exists());
        let csv_text = fs::read_to_string(&csv_file).unwrap();
        let lines: Vec<&str> = csv_text.lines().collect();
        assert!(lines.len() >= 2);
        assert_eq!(
            lines[0],
            "timestamp,epoch,batch,step_loss,ema_loss,early_halting_rate,gini,avg_hops,avg_energy,memory_density,active_subnodes,credit_volatility,temporal_affinity"
        );

        let row = lines[1];
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields[1], "1"); // epoch
        assert_eq!(fields[2], "42"); // batch_idx

        // Verify that all 10 metric fields are formatted with exactly 12 decimal places
        for field in &fields[3..13] {
            let parts: Vec<&str> = field.split('.').collect();
            assert_eq!(parts.len(), 2, "Field {} is missing decimal point", field);
            assert_eq!(
                parts[1].len(),
                12,
                "Field {} does not have 12 decimal places",
                field
            );
        }

        // Verify numeric values within single-precision epsilon
        assert!((fields[3].parse::<f32>().unwrap() - 2.75).abs() < 1e-6);
        assert!((fields[4].parse::<f32>().unwrap() - 2.8).abs() < 1e-6);

        let _ = fs::remove_file(log_file);
        let _ = fs::remove_file(csv_file);
    }

    #[test]
    fn test_eval_logger_and_csv_precision() {
        let tmp_dir = std::env::temp_dir();
        let log_file = tmp_dir.join("test_annp_eval_logger_run.log");
        let csv_file = tmp_dir.join("test_annp_eval_logger_run.csv");

        let _ = fs::remove_file(&log_file);
        let _ = fs::remove_file(&csv_file);

        let logger = AnnpLogger::new_eval(&tmp_dir, "eval_test", Some(&log_file));
        let eval_metrics = EvalBatchMetrics {
            next_token_mse: 0.045678912345,
            next_token_cossim: 0.987654321012,
            next_token_mae: 0.012345678901,
            nmse: 0.023456789012,
            psnr_db: 28.45678901,
            direct_recon_mse: 0.034567890123,
            internal_td_loss: 0.123456789012,
        };
        let model_metrics = BatchMetrics {
            avg_signal_energy: 1.1,
            utilization_gini: 0.12,
            early_halting_rate: 0.08,
            avg_hop_count: 2.4,
            avg_memory_density: 0.9,
            avg_subnodes: 1.2,
            avg_credit_volatility: 0.01,
            avg_temporal_affinity: 0.98,
        };

        logger.log_eval_step(10, 100, &eval_metrics, &model_metrics);

        assert!(log_file.exists());
        assert!(csv_file.exists());
        let csv_text = fs::read_to_string(&csv_file).unwrap();
        let lines: Vec<&str> = csv_text.lines().collect();
        assert!(lines.len() >= 2);
        assert_eq!(
            lines[0],
            "timestamp,batch,next_token_mse,next_token_cossim,next_token_mae,nmse,psnr_db,direct_recon_mse,internal_td_loss,early_halting_rate,gini,avg_hops,avg_energy,memory_density,active_subnodes"
        );

        let row = lines[1];
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields[1], "10"); // batch_idx
        assert!((fields[2].parse::<f32>().unwrap() - 0.045678912345).abs() < 1e-6);

        let _ = fs::remove_file(log_file);
        let _ = fs::remove_file(csv_file);
    }
}
