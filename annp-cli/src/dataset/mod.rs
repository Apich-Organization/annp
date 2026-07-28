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
use rand::Rng;
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

use std::path::PathBuf;

/// Zero-Memory-Overhead Streaming Dataset Loader for Massively Large Datasets
pub enum DatasetStream {
    ChunkedJson {
        chunk_paths: Vec<PathBuf>,
        current_chunk_idx: usize,
        current_tensors: Vec<Tensor>,
        cursor: usize,
        d_model: usize,
        device: Device,
    },
    Buffered {
        batches: Vec<Tensor>,
        cursor: usize,
    },
}

impl DatasetStream {
    pub fn new<P: AsRef<Path>>(
        path: P,
        format: DatasetFormat,
        d_model: usize,
        device: &Device,
    ) -> Result<Self> {
        let p = path.as_ref();
        if matches!(format, DatasetFormat::Json | DatasetFormat::Jsonl) && p.exists() {
            if let Ok((chunk_paths, _total_count)) = json_parser::split_and_cache_dataset(p, 200) {
                let mut stream = Self::ChunkedJson {
                    chunk_paths,
                    current_chunk_idx: 0,
                    current_tensors: Vec::new(),
                    cursor: 0,
                    d_model,
                    device: device.clone(),
                };
                let _ = stream.load_next_chunk()?;
                return Ok(stream);
            }
        }

        let batches = load_dataset(path, format, d_model, device)?;
        Ok(Self::Buffered { batches, cursor: 0 })
    }

    fn load_next_chunk(&mut self) -> Result<bool> {
        if let Self::ChunkedJson {
            chunk_paths,
            current_chunk_idx,
            current_tensors,
            cursor,
            d_model,
            device,
            ..
        } = self
        {
            if *current_chunk_idx >= chunk_paths.len() {
                return Ok(false);
            }
            let chunk_path = &chunk_paths[*current_chunk_idx];
            let tensors =
                json_parser::load_json_or_jsonl_dataset(chunk_path, true, *d_model, device)?;
            *current_tensors = tensors;
            *cursor = 0;
            *current_chunk_idx += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl Iterator for DatasetStream {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::ChunkedJson {
                cursor,
                current_tensors,
                ..
            } => {
                if *cursor < current_tensors.len() {
                    let tensor = current_tensors[*cursor].clone();
                    *cursor += 1;
                    return Some(Ok(tensor));
                }
            }
            Self::Buffered { batches, cursor } => {
                if *cursor < batches.len() {
                    let tensor = batches[*cursor].clone();
                    *cursor += 1;
                    return Some(Ok(tensor));
                }
                return None;
            }
        }

        if let Ok(true) = self.load_next_chunk() {
            if let Self::ChunkedJson {
                cursor,
                current_tensors,
                ..
            } = self
            {
                if *cursor < current_tensors.len() {
                    let tensor = current_tensors[*cursor].clone();
                    *cursor += 1;
                    return Some(Ok(tensor));
                }
            }
        }

        None
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
    let seq_len = 64;
    let num_batches = 128; // 128 rich synthetic batches
    let mut tensors = Vec::with_capacity(num_batches);
    let mut rng = rand::rng();

    // Multi-domain synthetic generator incorporating 4 structural dynamic modes:
    // Mode 0: Multi-frequency Harmonic Resonance with Phase Coupling
    // Mode 1: Long-range Associative Echo Memory Key-Value Pairs
    // Mode 2: Hierarchical Contextual Modulation (SwiGLU Non-linearity)
    // Mode 3: Zipfian Heavy-Tail Non-Linear Modulation & Dynamic Phase Drift

    let mut prev_memory_key = vec![0.0f32; d_model];

    for b in 0..num_batches {
        let mut flat = Vec::with_capacity(seq_len * d_model);
        let mut current_mode = b % 4;

        for t in 0..seq_len {
            let global_t = (b * seq_len + t) as f32;
            let t_norm = global_t * 0.05f32;

            // Transition Markov mode every 16 steps
            if t % 16 == 0 {
                current_mode = (current_mode + 1 + (rng.random_range(0..2))) % 4;
            }

            // Scatter key token at step 4 of each batch for long-range associative memory testing
            if t == 4 {
                for d in 0..d_model {
                    prev_memory_key[d] = ((d as f32 * 0.1).sin() + (t_norm * 0.3).cos()) * 1.5;
                }
            }

            for d in 0..d_model {
                let d_norm = d as f32 / d_model as f32;
                let d_val = d as f32 * 0.08f32;

                let val = match current_mode {
                    0 => {
                        // Multi-frequency Harmonic Resonance with Phase Coupling
                        let low_freq =
                            (t_norm * 0.2 + d_val).sin() + (t_norm * 0.05 - d_val * 0.5).cos();
                        let high_freq = 0.4f32 * (t_norm * 3.0 + d_val * 4.0).sin();
                        let coupling = 0.25f32 * (t_norm * 0.1 * d_val).sin();
                        low_freq + high_freq + coupling
                    }
                    1 => {
                        // Long-Range Associative Memory Retrieval (Step t=48 echoes Key from t=4)
                        if t >= 48 && t <= 52 {
                            let decay = (-((t as i32 - 48) as f32 * 0.5)).exp();
                            prev_memory_key[d] * decay + 0.1 * (t_norm + d_val).sin()
                        } else {
                            0.5 * (t_norm * 0.4 + d_val * 2.0).cos()
                        }
                    }
                    2 => {
                        // Hierarchical Nested Syntax (SwiGLU Modulation)
                        let gate = (t_norm * 0.5 + d_val).sin();
                        let up = (t_norm * 0.8 - d_val * 1.2).cos();
                        let swish = gate / (1.0 + (-gate).exp());
                        swish * up * (1.0 + 0.5 * d_norm)
                    }
                    _ => {
                        // Zipfian Non-Linear Burst with Dynamic Phase Drift
                        let zipf_factor = 1.0 / (1.0 + (d as f32 * 0.05));
                        let carrier = (t_norm * 1.5 + d_norm * 6.28).sin();
                        let tanh_mod = (carrier * 2.0).tanh();
                        tanh_mod * zipf_factor * 1.8
                    }
                };

                flat.push(val);
            }
        }
        let t = Tensor::from_vec(flat, (seq_len, d_model), device)?;
        tensors.push(t);
    }

    Ok(tensors)
}
