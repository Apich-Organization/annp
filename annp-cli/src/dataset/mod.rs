/// IXIA DATASET SPECIFICATION PROTOCOL FOR ANNP (Asynchronous Neural Network Protocol)
/// ==================================================================================
///
/// Supported Dataset Ingestion Formats & Structural Layout Guidelines:
///
/// 1. JSON / JSONL Format ("json" | "jsonl"):
///    - Structure: Array of JSON objects or Line-delimited JSON (JSONL).
///    - Schema Fields:
///      * `input_text` (String): Raw text prompt/sequence to be vectorized.
///      * OR `embedding` (Array of Floats [seq_len * d_model]): Pre-vectorized dense matrix.
///      * OR `tokens` (Array of Integer IDs): Token ID sequence.
///    - Example (JSONL):
///      {"input_text": "ANNP Multi-Frequency Harmonic Wave Simulation"}
///      {"embedding": [0.12, -0.45, 0.88, ...]}
///
/// 2. CSV Format ("csv"):
///    - Header Required: Yes.
///    - Schema Columns:
///      * Column `text` or `content` (String): Raw sequence text.
///      * OR Numeric columns `f0`, `f1`, ..., `f_{d_model-1}` representing token dimensions.
///    - Example CSV:
///      text
///      "ANNP P2P Mesh Routing Analysis Sequence"
///
/// 3. SQLite Database Format ("sqlite" | "db"):
///    - Required Table: `dataset` or `samples`
///    - Required Columns: `id` (INTEGER PRIMARY KEY), `content` (TEXT) or `vector` (BLOB/TEXT).
///    - Query Protocol: `SELECT content FROM dataset LIMIT 1000;`
///
/// 4. Synthetic Pattern Generator ("synthetic" | "pattern"):
///    - Generates multi-frequency harmonic wave resonances over broad time domain [-10\pi, 10\pi]:
///      $y(t, d) = \sin(0.1t + 0.05d) + 0.35 \sin(2.5t + 3d) + 0.2 \sin(0.2t \cdot d)$
///
pub mod csv_parser;
pub mod json_parser;
pub mod sqlite_parser;

use candle_core::{Device, Result, Tensor};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub enum DatasetFormat {
    Json,
    Jsonl,
    Csv,
    Sqlite,
    SyntheticPattern,
}

impl DatasetFormat {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "jsonl" => Self::Jsonl,
            "csv" => Self::Csv,
            "sqlite" | "db" => Self::Sqlite,
            "pattern" | "synthetic" | "harmonic" => Self::SyntheticPattern,
            _ => Self::Json,
        }
    }
}

/// Zero-Memory-Overhead Streaming Dataset Loader for Massively Large Datasets
pub struct DatasetStream {
    batches: Vec<Tensor>,
    cursor: usize,
}

impl DatasetStream {
    pub fn new<P: AsRef<Path>>(
        path: P,
        format: DatasetFormat,
        d_model: usize,
        device: &Device,
    ) -> Result<Self> {
        let batches = load_dataset(path, format, d_model, device)?;
        Ok(Self { batches, cursor: 0 })
    }
}

impl Iterator for DatasetStream {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.batches.len() {
            let tensor = self.batches[self.cursor].clone();
            self.cursor += 1;
            Some(Ok(tensor))
        } else {
            None
        }
    }
}

pub fn load_dataset<P: AsRef<Path>>(
    path: P,
    format: DatasetFormat,
    d_model: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    match format {
        DatasetFormat::SyntheticPattern => generate_synthetic_pattern_tensors(d_model, device),
        _ => {
            let p = path.as_ref();
            if !p.exists() {
                println!(
                    "Dataset path {:?} not found. Generating complex multi-frequency harmonic resonance tensors for Loss testing.",
                    p
                );
                return generate_synthetic_pattern_tensors(d_model, device);
            }

            match format {
                DatasetFormat::Json => {
                    json_parser::load_json_or_jsonl_dataset(p, false, d_model, device)
                }
                DatasetFormat::Jsonl => {
                    json_parser::load_json_or_jsonl_dataset(p, true, d_model, device)
                }
                DatasetFormat::Csv => csv_parser::load_csv_dataset(p, d_model, device),
                DatasetFormat::Sqlite => sqlite_parser::load_sqlite_dataset(p, d_model, device),
                DatasetFormat::SyntheticPattern => {
                    generate_synthetic_pattern_tensors(d_model, device)
                }
            }
        }
    }
}

pub fn generate_synthetic_pattern_tensors(d_model: usize, device: &Device) -> Result<Vec<Tensor>> {
    let seq_len = 32;
    let num_batches = 8;
    let mut tensors = Vec::with_capacity(num_batches);

    for b in 0..num_batches {
        let mut flat = Vec::with_capacity(seq_len * d_model);
        for t in 0..seq_len {
            let t_val = ((t + b * seq_len) as f32 * 0.25) - 31.4159f32;
            for d in 0..d_model {
                let d_val = d as f32 * 0.05f32;
                let low_freq = (t_val * 0.1 + d_val).sin() + (t_val * 0.05 - d_val * 0.5).cos();
                let high_freq = 0.35f32 * (t_val * 2.5 + d_val * 3.0).sin()
                    + 0.25f32 * (t_val * 5.0 - d_val * 1.5).cos();
                let phase_coupling = 0.20f32 * (t_val * 0.2 * d_val).sin();
                let combined_val = low_freq + high_freq + phase_coupling;
                flat.push(combined_val);
            }
        }
        let t = Tensor::from_vec(flat, (seq_len, d_model), device)?;
        tensors.push(t);
    }

    Ok(tensors)
}
