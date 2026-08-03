use crate::config::AnnpTomlConfig;
use crate::dataset::{DatasetFormat, DatasetStream};
use candle_core::Device;
use clap::Args;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct ExportDatasetArgs {
    /// Dataset name or generator ("markov", "periodic", "harmonic", "recall", "chaos", "noise", "random", "pattern")
    #[arg(short = 's', long = "dataset", default_value = "markov")]
    pub dataset: String,

    /// Number of sequence batches to export
    #[arg(short = 'b', long = "batches", default_value_t = 1000)]
    pub batches: usize,

    /// Output file path (e.g. "dataset_markov.jsonl")
    #[arg(short = 'o', long = "output")]
    pub output: PathBuf,

    /// Output format ("jsonl", "json", "csv", "bin") (auto-inferred from extension if not specified)
    #[arg(short = 'f', long = "format")]
    pub format: Option<String>,

    /// Embedding dimension d_model (default: from config or 64)
    #[arg(long = "d-model")]
    pub d_model: Option<usize>,

    /// Target device ("cpu", "cuda", "auto")
    #[arg(short = 'd', long = "device", default_value = "cpu")]
    pub device: String,
}

pub fn execute_export_dataset(args: ExportDatasetArgs) -> Result<(), Box<dyn std::error::Error>> {
    let toml_cfg = AnnpTomlConfig::load_from_file("annp_config.toml").ok();
    let d_model = args
        .d_model
        .or_else(|| toml_cfg.as_ref().map(|c| c.to_core_config().d_model()))
        .unwrap_or(64);

    let device = match args.device.as_str() {
        "cuda" => Device::new_cuda(0).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    };

    let dataset_format = DatasetFormat::parse(&args.dataset);
    println!(
        "Exporting dataset '{}' (Format: {:?}, d_model: {}, batches: {}) to {:?}",
        args.dataset, dataset_format, d_model, args.batches, args.output
    );

    let (stream, _total_available) = DatasetStream::new_with_batch_count(
        &args.dataset,
        dataset_format,
        d_model,
        args.batches,
        &device,
    )?;

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let out_format = args.format.map(|f| f.to_lowercase()).unwrap_or_else(|| {
        args.output
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("jsonl")
            .to_lowercase()
    });

    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);

    let mut exported_batches = 0;
    let mut total_tokens = 0;

    match out_format.as_str() {
        "csv" => {
            // Write CSV header
            write!(writer, "batch_idx,step")?;
            for d in 0..d_model {
                write!(writer, ",f{}", d)?;
            }
            writeln!(writer)?;

            for tensor_res in stream {
                let tensor = tensor_res?;
                let (_seq_len, _dim) = tensor.dims2()?;
                let values: Vec<Vec<f32>> = tensor.to_vec2()?;

                for (step, token_vec) in values.into_iter().enumerate() {
                    write!(writer, "{},{}", exported_batches, step)?;
                    for v in token_vec {
                        write!(writer, ",{:.12}", v)?;
                    }
                    writeln!(writer)?;
                    total_tokens += 1;
                }

                exported_batches += 1;
                if exported_batches >= args.batches {
                    break;
                }
            }
        }
        "json" => {
            writeln!(writer, "[")?;
            let mut first = true;
            for tensor_res in stream {
                let tensor = tensor_res?;
                let (seq_len, dim) = tensor.dims2()?;
                let values: Vec<Vec<f32>> = tensor.to_vec2()?;
                total_tokens += seq_len;

                if !first {
                    writeln!(writer, ",")?;
                }
                first = false;

                let row = serde_json::json!({
                    "batch_idx": exported_batches,
                    "seq_len": seq_len,
                    "d_model": dim,
                    "values": values,
                });
                write!(writer, "  {}", serde_json::to_string(&row)?)?;

                exported_batches += 1;
                if exported_batches >= args.batches {
                    break;
                }
            }
            writeln!(writer, "\n]")?;
        }
        "bin" => {
            for tensor_res in stream {
                let tensor = tensor_res?;
                let (seq_len, dim) = tensor.dims2()?;
                let values: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
                total_tokens += seq_len;

                writer.write_all(&(seq_len as u32).to_le_bytes())?;
                writer.write_all(&(dim as u32).to_le_bytes())?;
                for v in values {
                    writer.write_all(&v.to_le_bytes())?;
                }

                exported_batches += 1;
                if exported_batches >= args.batches {
                    break;
                }
            }
        }
        _ => {
            // Default: jsonl format
            for tensor_res in stream {
                let tensor = tensor_res?;
                let (seq_len, dim) = tensor.dims2()?;
                let values: Vec<Vec<f32>> = tensor.to_vec2()?;
                total_tokens += seq_len;

                let row = serde_json::json!({
                    "batch_idx": exported_batches,
                    "seq_len": seq_len,
                    "d_model": dim,
                    "values": values,
                });
                writeln!(writer, "{}", serde_json::to_string(&row)?)?;

                exported_batches += 1;
                if exported_batches >= args.batches {
                    break;
                }
            }
        }
    }

    writer.flush()?;
    let file_size = fs::metadata(&args.output)?.len();

    println!(
        "Successfully exported {} batches ({} tokens, {:.2} MB) to {:?}",
        exported_batches,
        total_tokens,
        file_size as f64 / (1024.0 * 1024.0),
        args.output
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_dataset_csv_and_json() -> Result<(), Box<dyn std::error::Error>> {
        let temp_dir =
            std::env::temp_dir().join(format!("annp_test_export_{}", std::process::id()));
        fs::create_dir_all(&temp_dir)?;
        let csv_path = temp_dir.join("markov_export.csv");
        let json_path = temp_dir.join("periodic_export.json");

        let args_csv = ExportDatasetArgs {
            dataset: "markov".into(),
            output: csv_path.clone(),
            format: Some("csv".into()),
            batches: 5,
            d_model: Some(16),
            device: "cpu".into(),
        };

        execute_export_dataset(args_csv)?;
        assert!(csv_path.exists());
        let csv_content = fs::read_to_string(&csv_path)?;
        let lines: Vec<&str> = csv_content.lines().collect();
        assert_eq!(lines.len(), 1 + 5 * 64, "CSV must have header + 5*64 rows");
        assert!(lines[0].starts_with("batch_idx,step,f0"));

        let args_json = ExportDatasetArgs {
            dataset: "periodic".into(),
            output: json_path.clone(),
            format: Some("json".into()),
            batches: 3,
            d_model: Some(16),
            device: "cpu".into(),
        };

        execute_export_dataset(args_json)?;
        assert!(json_path.exists());
        let json_content = fs::read_to_string(&json_path)?;
        let parsed: serde_json::Value = serde_json::from_str(&json_content)?;
        assert_eq!(parsed.as_array().unwrap().len(), 3);

        let _ = fs::remove_dir_all(temp_dir);
        Ok(())
    }
}
