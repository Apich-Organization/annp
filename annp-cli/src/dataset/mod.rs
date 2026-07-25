pub mod csv_parser;
pub mod json_parser;
pub mod sqlite_parser;

use candle_core::{Device, Result, Tensor};
use std::path::Path;

pub enum DatasetFormat {
    Json,
    Jsonl,
    Csv,
    Sqlite,
}

impl DatasetFormat {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "jsonl" => Self::Jsonl,
            "csv" => Self::Csv,
            "sqlite" | "db" => Self::Sqlite,
            _ => Self::Json,
        }
    }
}

pub fn load_dataset<P: AsRef<Path>>(
    path: P,
    format: DatasetFormat,
    d_model: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let p = path.as_ref();
    if !p.exists() {
        // Fallback synthetic dataset generation if file does not exist yet
        println!("Dataset path {:?} not found. Generating synthetic training tensors for PoC execution.", p);
        return Ok(vec![
            Tensor::randn(0.0f32, 1.0f32, (8, d_model), device)?,
            Tensor::randn(0.0f32, 1.0f32, (8, d_model), device)?,
        ]);
    }

    match format {
        DatasetFormat::Json => json_parser::load_json_or_jsonl_dataset(p, false, d_model, device),
        DatasetFormat::Jsonl => json_parser::load_json_or_jsonl_dataset(p, true, d_model, device),
        DatasetFormat::Csv => csv_parser::load_csv_dataset(p, d_model, device),
        DatasetFormat::Sqlite => sqlite_parser::load_sqlite_dataset(p, d_model, device),
    }
}
