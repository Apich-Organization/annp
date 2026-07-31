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
    RandomTokens,
}

impl DatasetFormat {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "jsonl" => Self::Jsonl,
            "csv" => Self::Csv,
            "sqlite" | "db" => Self::Sqlite,
            "pattern" | "synthetic" | "harmonic" => Self::SyntheticPattern,
            "random" | "randomtokens" => Self::RandomTokens,
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
    ChunkedCsv {
        chunk_paths: Vec<PathBuf>,
        current_chunk_idx: usize,
        current_tensors: Vec<Tensor>,
        cursor: usize,
        d_model: usize,
        device: Device,
    },
    ChunkedSqlite {
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
    ) -> Result<(Self, usize)> {
        let p = path.as_ref();
        if matches!(format, DatasetFormat::Json | DatasetFormat::Jsonl) && p.exists() {
            if let Ok((chunk_paths, _total_count)) = json_parser::split_and_cache_dataset(p, 8192) {
                let mut stream = Self::ChunkedJson {
                    chunk_paths,
                    current_chunk_idx: 0,
                    current_tensors: Vec::new(),
                    cursor: 0,
                    d_model,
                    device: device.clone(),
                };
                let _ = stream.load_next_chunk()?;
                return Ok((stream, _total_count));
            }
        } else if matches!(format, DatasetFormat::Csv) && p.exists() {
            if let Ok((chunk_paths, _total_count)) = csv_parser::split_and_cache_dataset(p, 8192) {
                let mut stream = Self::ChunkedCsv {
                    chunk_paths,
                    current_chunk_idx: 0,
                    current_tensors: Vec::new(),
                    cursor: 0,
                    d_model,
                    device: device.clone(),
                };
                let _ = stream.load_next_chunk()?;
                return Ok((stream, _total_count));
            }
        } else if matches!(format, DatasetFormat::Sqlite)
            && p.exists()
            && let Ok((chunk_paths, _total_count)) = sqlite_parser::split_and_cache_dataset(p, 8192)
        {
            let mut stream = Self::ChunkedSqlite {
                chunk_paths,
                current_chunk_idx: 0,
                current_tensors: Vec::new(),
                cursor: 0,
                d_model,
                device: device.clone(),
            };
            let _ = stream.load_next_chunk()?;
            return Ok((stream, _total_count));
        }

        let batches = load_dataset(path, format, d_model, device)?;
        let total = batches.len();
        Ok((Self::Buffered { batches, cursor: 0 }, total))
    }

    fn load_next_chunk(&mut self) -> Result<bool> {
        match self {
            Self::ChunkedJson {
                chunk_paths,
                current_chunk_idx,
                current_tensors,
                cursor,
                d_model,
                device,
            } => {
                if *current_chunk_idx >= chunk_paths.len() {
                    return Ok(false);
                }
                let chunk_path = &chunk_paths[*current_chunk_idx];
                *current_tensors =
                    json_parser::load_json_or_jsonl_dataset(chunk_path, true, *d_model, device)?;
                *cursor = 0;
                *current_chunk_idx += 1;
                Ok(true)
            }
            Self::ChunkedCsv {
                chunk_paths,
                current_chunk_idx,
                current_tensors,
                cursor,
                d_model,
                device,
            } => {
                if *current_chunk_idx >= chunk_paths.len() {
                    return Ok(false);
                }
                let chunk_path = &chunk_paths[*current_chunk_idx];
                *current_tensors = csv_parser::load_csv_dataset(chunk_path, *d_model, device)?;
                *cursor = 0;
                *current_chunk_idx += 1;
                Ok(true)
            }
            Self::ChunkedSqlite {
                chunk_paths,
                current_chunk_idx,
                current_tensors,
                cursor,
                d_model,
                device,
            } => {
                if *current_chunk_idx >= chunk_paths.len() {
                    return Ok(false);
                }
                let chunk_path = &chunk_paths[*current_chunk_idx];
                *current_tensors = sqlite_parser::load_sqlite_chunk(chunk_path, *d_model, device)?;
                *cursor = 0;
                *current_chunk_idx += 1;
                Ok(true)
            }
            _ => Ok(false),
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
            }
            | Self::ChunkedCsv {
                cursor,
                current_tensors,
                ..
            }
            | Self::ChunkedSqlite {
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
            match self {
                Self::ChunkedJson {
                    cursor,
                    current_tensors,
                    ..
                }
                | Self::ChunkedCsv {
                    cursor,
                    current_tensors,
                    ..
                }
                | Self::ChunkedSqlite {
                    cursor,
                    current_tensors,
                    ..
                } if *cursor < current_tensors.len() => {
                    let tensor = current_tensors[*cursor].clone();
                    *cursor += 1;
                    return Some(Ok(tensor));
                }
                _ => {}
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
        DatasetFormat::RandomTokens => generate_random_tokens_tensors(d_model, device),
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
                DatasetFormat::RandomTokens => generate_random_tokens_tensors(d_model, device),
            }
        }
    }
}

pub fn generate_random_tokens_tensors(d_model: usize, device: &Device) -> Result<Vec<Tensor>> {
    use crate::tokenizer::AnnpTokenizer;
    let tokenizer = AnnpTokenizer::load_from_file("tokenizer.model");

    let seq_len = 64;
    let num_batches = 128;
    let mut tensors = Vec::with_capacity(num_batches);
    let mut rng = rand::rng();

    for _ in 0..num_batches {
        // Generate random sequence of token IDs
        let mut text_parts = Vec::new();
        for _ in 0..seq_len {
            let token_id: u32 = rng.random_range(1..32000);
            let s = tokenizer.decode(&[token_id]);
            // Filter out empty or whitespace to ensure they count as valid tokens
            if !s.trim().is_empty() {
                text_parts.push(s);
            } else {
                text_parts.push(format!("t{}", token_id));
            }
        }
        let text = text_parts.join(" ");
        let tensor = tokenizer.encode_to_tensor(&text, d_model, device)?;
        // If the tokenizer output seq_len doesn't match perfectly, that's fine, we take the subset or let it be
        tensors.push(tensor);
    }
    println!(
        "Generated {} batches of Random Token data for baseline testing.",
        num_batches
    );
    Ok(tensors)
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
                for (d, val) in prev_memory_key.iter_mut().enumerate() {
                    *val = ((d as f32 * 0.1).sin() + (t_norm * 0.3).cos()) * 1.5;
                }
            }

            for (d, &prev_key_val) in prev_memory_key.iter().enumerate() {
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
                        if (48..=52).contains(&t) {
                            let decay = (-((t as i32 - 48) as f32 * 0.5)).exp();
                            prev_key_val * decay + 0.1 * (t_norm + d_val).sin()
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
                        let carrier = (t_norm * 1.5 + d_norm * std::f32::consts::TAU).sin();
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
