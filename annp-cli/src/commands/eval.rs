use crate::checkpoint::ModelCheckpoint;
use crate::commands::train::select_device;
use crate::dataset::{DatasetFormat, DatasetStream};
use crate::logger::{AnnpLogger, AssociativeRecallReport, EvalBatchMetrics, EvalSummaryReport};
use annp_model::ANNPModel;
use clap::Args;
use std::fs;
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Path to model checkpoint file (.annpb or .json)
    #[arg(short = 'c', long = "checkpoint")]
    pub checkpoint: PathBuf,

    /// Dataset name or path ("markov", "periodic", "harmonic", "recall", "chaos", "noise", "random", "pattern", or file path)
    #[arg(short = 's', long = "dataset", default_value = "markov")]
    pub dataset: String,

    /// Number of evaluation batches to run (default: 1000)
    #[arg(short = 'b', long = "batches", default_value_t = 1000)]
    pub batches: usize,

    /// Directory for evaluation logs and CSV output
    #[arg(short = 'l', long = "log-dir", default_value = "logs/eval")]
    pub log_dir: PathBuf,

    /// Sampling interval for logging and CSV dumping (in batches, default: 2 matching training)
    #[arg(long = "sample-interval", default_value_t = 2)]
    pub sample_interval: usize,

    /// Enable online test-time plasticity (default: false, frozen weights for pure inference evaluation)
    #[arg(long = "plasticity", default_value_t = false)]
    pub plasticity: bool,

    /// Optional learning rate if plasticity is enabled (default: 0.001)
    #[arg(long = "lr", default_value_t = 0.001)]
    pub lr: f32,

    /// Optional JSON file path to dump machine-readable evaluation report
    #[arg(long = "json-output")]
    pub json_output: Option<PathBuf>,

    /// Target compute device ("cpu", "cuda", "auto")
    #[arg(short = 'd', long = "device", default_value = "cpu")]
    pub device: String,
}

pub fn execute_eval(args: EvalArgs) -> Result<(), Box<dyn std::error::Error>> {
    let logger = AnnpLogger::new_eval(&args.log_dir, "eval", None);

    logger.log(
        "INIT",
        &format!("Loading ANNP Model Checkpoint from: {:?}", args.checkpoint),
    );
    let ckpt = ModelCheckpoint::load(&args.checkpoint)?;

    let (device, use_cuda) = select_device(&args.device);

    let num_shards = ckpt.config.num_shards;
    let d_model = ckpt.config.d_model();
    let num_nodes = ckpt.nodes.len();

    let mut model = ANNPModel::new_with_cuda(
        num_nodes,
        num_shards,
        ckpt.config.clone(),
        device.clone(),
        use_cuda,
    );

    ckpt.apply_to_model(&mut model);
    model.is_training = args.plasticity;

    logger.log(
        "SYSTEM",
        "=== Starting ANNP External Task Fidelity Evaluation ===",
    );
    logger.log(
        "SYSTEM",
        &format!(
            "Model Topology: {} nodes ({}x{}), d_head={}, d_model={}, k_neighbors={} | Plasticity: {}",
            num_nodes,
            model.config.mesh_rows,
            model.config.mesh_cols,
            model.config.d_head,
            d_model,
            model.config.k_neighbors,
            if args.plasticity { "ENABLED" } else { "FROZEN (Pure Inference)" }
        ),
    );

    let dataset_format = DatasetFormat::parse(&args.dataset);
    logger.log(
        "DATASET",
        &format!(
            "Evaluation Dataset: '{}' (Format: {:?}, Batches: {})",
            args.dataset, dataset_format, args.batches
        ),
    );

    let (stream, _total_available) = DatasetStream::new_with_batch_count(
        &args.dataset,
        dataset_format,
        d_model,
        args.batches,
        &device,
    )?;

    let eval_lr = if args.plasticity { Some(args.lr) } else { None };

    let mut total_evaluated_batches = 0usize;
    let mut all_next_mse = Vec::with_capacity(args.batches);
    let mut all_next_cossim = Vec::with_capacity(args.batches);
    let mut all_next_mae = Vec::with_capacity(args.batches);
    let mut all_td_loss = Vec::with_capacity(args.batches);
    let mut all_recon_mse = Vec::with_capacity(args.batches);

    let mut sum_nmse_num = 0.0f64;
    let mut sum_nmse_den = 0.0f64;
    let mut max_observed_abs_val = 1e-6f64;

    let mut sum_early_halting = 0.0f64;
    let mut sum_hops = 0.0f64;
    let mut sum_subnodes = 0.0f64;

    // Associative recall tracking
    let is_recall_benchmark = dataset_format == DatasetFormat::AssociativeRecall;
    let mut recall_key_mse_sum = 0.0f64;
    let mut recall_key_cossim_sum = 0.0f64;
    let mut recall_key_success_count = 0usize;
    let mut recall_key_total_count = 0usize;

    let sample_interval = args.sample_interval.max(1);

    for tensor_res in stream {
        let input_embeddings = tensor_res?;
        let (seq_len, _dim) = input_embeddings.dims2()?;

        // Reset temporal routing state before sequence inference
        model.reset_state();

        let ground_truth_vecs: Vec<Vec<f32>> = input_embeddings.to_vec2()?;
        let mut predicted_emissions: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let mut batch_td_loss = 0.0f32;

        for t in 0..seq_len {
            let single_token = input_embeddings.narrow(0, t, 1)?;
            let (out_tensor, step_loss) = model.forward(&single_token, t, eval_lr)?;
            batch_td_loss += step_loss;

            let out_vec = out_tensor.flatten_all()?.to_vec1::<f32>()?;
            predicted_emissions.push(out_vec);
        }

        batch_td_loss /= seq_len as f32;

        // Compute External Task Fidelity Metrics
        // 1. Next-token prediction metrics: compare emission at t with ground-truth at t+1
        let mut next_mse_acc = 0.0f64;
        let mut next_mae_acc = 0.0f64;
        let mut next_cossim_acc = 0.0f64;
        let mut nmse_num = 0.0f64;
        let mut nmse_den = 0.0f64;
        let mut batch_max_abs = 1e-6f64;

        let num_next_tokens = if seq_len > 1 { seq_len - 1 } else { 1 };

        for t in 0..(seq_len.saturating_sub(1)) {
            let pred = &predicted_emissions[t];
            let target = &ground_truth_vecs[t + 1];

            let mut dot_prod = 0.0f64;
            let mut norm_p = 0.0f64;
            let mut norm_t = 0.0f64;
            let mut se = 0.0f64;
            let mut ae = 0.0f64;

            for d in 0..d_model {
                let p_val = pred[d] as f64;
                let t_val = target[d] as f64;
                let diff = t_val - p_val;

                se += diff * diff;
                ae += diff.abs();
                dot_prod += p_val * t_val;
                norm_p += p_val * p_val;
                norm_t += t_val * t_val;

                if t_val.abs() > batch_max_abs {
                    batch_max_abs = t_val.abs();
                }
            }

            let token_mse = se / (d_model as f64);
            let token_mae = ae / (d_model as f64);
            let token_cossim = if norm_p > 1e-12 && norm_t > 1e-12 {
                dot_prod / (norm_p.sqrt() * norm_t.sqrt())
            } else {
                0.0
            };

            next_mse_acc += token_mse;
            next_mae_acc += token_mae;
            next_cossim_acc += token_cossim;
            nmse_num += se;
            nmse_den += norm_t;
        }

        if batch_max_abs > max_observed_abs_val {
            max_observed_abs_val = batch_max_abs;
        }

        let batch_next_mse = (next_mse_acc / num_next_tokens as f64) as f32;
        let batch_next_mae = (next_mae_acc / num_next_tokens as f64) as f32;
        let batch_next_cossim = (next_cossim_acc / num_next_tokens as f64) as f32;
        let batch_nmse = (if nmse_den > 1e-12 {
            nmse_num / nmse_den
        } else {
            0.0
        }) as f32;
        let batch_psnr = if batch_next_mse > 1e-12 {
            (10.0 * ((batch_max_abs * batch_max_abs) / (batch_next_mse as f64)).log10()) as f32
        } else {
            100.0f32
        };

        sum_nmse_num += nmse_num;
        sum_nmse_den += nmse_den;

        // 2. Direct Reconstruction MSE: compare emission at t with input at t
        let mut recon_se = 0.0f64;
        for t in 0..seq_len {
            let pred = &predicted_emissions[t];
            let gt = &ground_truth_vecs[t];
            for d in 0..d_model {
                let diff = gt[d] as f64 - pred[d] as f64;
                recon_se += diff * diff;
            }
        }
        let batch_recon_mse = (recon_se / (seq_len * d_model) as f64) as f32;

        // 3. Associative Recall evaluation: Key at t=4..7, Predictive Recall Cue at t=44..47
        if is_recall_benchmark && seq_len >= 48 {
            for key_idx in 0..4 {
                let key_t = 4 + key_idx;
                // At t=44 (trigger cue pulse), model predicts key 0 (which appears as target at t=45).
                // At t=44+k, predicted_emissions[44+k] predicts key k without seeing key k as input.
                let recall_t = 44 + key_idx;

                let key_gt = &ground_truth_vecs[key_t];
                let recalled_pred = &predicted_emissions[recall_t];

                let mut se = 0.0f64;
                let mut dot_prod = 0.0f64;
                let mut norm_k = 0.0f64;
                let mut norm_r = 0.0f64;

                for d in 0..d_model {
                    let k = key_gt[d] as f64;
                    let r = recalled_pred[d] as f64;
                    let diff = k - r;
                    se += diff * diff;
                    dot_prod += k * r;
                    norm_k += k * k;
                    norm_r += r * r;
                }

                let key_mse = se / d_model as f64;
                let key_cossim = if norm_k > 1e-12 && norm_r > 1e-12 {
                    dot_prod / (norm_k.sqrt() * norm_r.sqrt())
                } else {
                    0.0
                };

                recall_key_mse_sum += key_mse;
                recall_key_cossim_sum += key_cossim;
                recall_key_total_count += 1;
                if key_cossim >= 0.9 {
                    recall_key_success_count += 1;
                }
            }
        }

        let model_metrics = model.extract_batch_metrics();

        let eval_batch_metrics = EvalBatchMetrics {
            next_token_mse: batch_next_mse,
            next_token_cossim: batch_next_cossim,
            next_token_mae: batch_next_mae,
            nmse: batch_nmse,
            psnr_db: batch_psnr,
            direct_recon_mse: batch_recon_mse,
            internal_td_loss: batch_td_loss,
        };

        total_evaluated_batches += 1;
        all_next_mse.push(batch_next_mse);
        all_next_cossim.push(batch_next_cossim);
        all_next_mae.push(batch_next_mae);
        all_td_loss.push(batch_td_loss);
        all_recon_mse.push(batch_recon_mse);

        sum_early_halting += model_metrics.early_halting_rate as f64;
        sum_hops += model_metrics.avg_hop_count as f64;
        sum_subnodes += model_metrics.avg_subnodes as f64;

        if total_evaluated_batches.is_multiple_of(sample_interval)
            || total_evaluated_batches == args.batches
        {
            logger.log_eval_step(
                total_evaluated_batches,
                args.batches,
                &eval_batch_metrics,
                &model_metrics,
            );
        }

        if total_evaluated_batches >= args.batches {
            break;
        }
    }

    if total_evaluated_batches == 0 {
        return Err("No evaluation batches were processed.".into());
    }

    let n = total_evaluated_batches as f64;

    // Compute rigorous statistical distributions
    let mean_f32 = |v: &[f32]| -> f32 {
        if v.is_empty() {
            0.0
        } else {
            (v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64) as f32
        }
    };

    let std_f32 = |v: &[f32], mean: f32| -> f32 {
        if v.len() <= 1 {
            0.0
        } else {
            let var = v
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean as f64;
                    d * d
                })
                .sum::<f64>()
                / (v.len() as f64);
            var.sqrt() as f32
        }
    };

    let percentile_f32 = |v: &mut [f32], pct: f64| -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((v.len() as f64 * pct).round() as usize).min(v.len() - 1);
        v[idx]
    };

    let avg_next_mse = mean_f32(&all_next_mse);
    let std_next_mse = std_f32(&all_next_mse, avg_next_mse);
    let mut sorted_mse = all_next_mse.clone();
    sorted_mse.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_next_mse = sorted_mse.first().copied().unwrap_or(0.0);
    let max_next_mse = sorted_mse.last().copied().unwrap_or(0.0);
    let median_next_mse = percentile_f32(&mut sorted_mse, 0.50);
    let p95_next_mse = percentile_f32(&mut sorted_mse, 0.95);

    let avg_next_cossim = mean_f32(&all_next_cossim);
    let std_next_cossim = std_f32(&all_next_cossim, avg_next_cossim);

    let avg_next_mae = mean_f32(&all_next_mae);
    let avg_recon_mse = mean_f32(&all_recon_mse);

    let avg_td_loss = mean_f32(&all_td_loss);
    let std_td_loss = std_f32(&all_td_loss, avg_td_loss);

    let grand_nmse = if sum_nmse_den > 1e-12 {
        (sum_nmse_num / sum_nmse_den) as f32
    } else {
        0.0
    };

    let grand_psnr = if avg_next_mse > 1e-12 {
        (10.0 * ((max_observed_abs_val * max_observed_abs_val) / (avg_next_mse as f64)).log10())
            as f32
    } else {
        100.0
    };

    let td_to_mse_ratio = if avg_next_mse > 1e-12 {
        avg_td_loss / avg_next_mse
    } else {
        1.0
    };

    let associative_recall = if is_recall_benchmark && recall_key_total_count > 0 {
        Some(AssociativeRecallReport {
            key_recall_mse: (recall_key_mse_sum / recall_key_total_count as f64) as f32,
            key_recall_cossim: (recall_key_cossim_sum / recall_key_total_count as f64) as f32,
            retrieval_accuracy: (recall_key_success_count as f32) / (recall_key_total_count as f32),
        })
    } else {
        None
    };

    let summary_report = EvalSummaryReport {
        checkpoint_path: Some(args.checkpoint.display().to_string()),
        dataset_name: args.dataset.clone(),
        total_batches: total_evaluated_batches,
        next_token_mse_mean: avg_next_mse,
        next_token_mse_std: std_next_mse,
        next_token_mse_median: median_next_mse,
        next_token_mse_min: min_next_mse,
        next_token_mse_max: max_next_mse,
        next_token_mse_p95: p95_next_mse,
        next_token_cossim_mean: avg_next_cossim,
        next_token_cossim_std: std_next_cossim,
        next_token_mae_mean: avg_next_mae,
        nmse: grand_nmse,
        psnr_db: grand_psnr,
        direct_recon_mse_mean: avg_recon_mse,
        internal_td_loss_mean: avg_td_loss,
        internal_td_loss_std: std_td_loss,
        td_to_mse_ratio,
        early_halting_rate: (sum_early_halting / n) as f32,
        avg_hops: (sum_hops / n) as f32,
        active_subnodes: (sum_subnodes / n) as f32,
        associative_recall,
    };

    logger.log_eval_summary(&summary_report);

    if let Some(ref json_path) = args.json_output {
        if let Some(parent) = json_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let json_str = serde_json::to_string_pretty(&summary_report)?;
        fs::write(json_path, json_str)?;
        println!("Saved JSON evaluation report to: {:?}", json_path);
    }

    Ok(())
}
