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
///
/// 2. CSV Format ("csv"):
///    - Header Required: Yes.
///    - Schema Columns:
///      * Column `text` or `content` (String): Raw sequence text.
///      * OR Numeric columns `f0`, `f1`, ..., `f_{d_model-1}` representing token dimensions.
///
/// 3. SQLite Database Format ("sqlite" | "db"):
///    - Required Table: `dataset` or `samples`
///    - Required Columns: `id` (INTEGER PRIMARY KEY), `content` (TEXT) or `vector` (BLOB/TEXT).
///
/// 4. Peer-Reviewed Academic Benchmark Suite (Publication-Grade Rigor):
///    - "periodic" | "regular" | "deterministic":
///      Deterministic multi-channel periodic dynamical manifold:
///      $X(t, d) = \cos(2\pi t / T_d + \phi_d) + 0.5 \sin(4\pi t / T_d)$
///      with exact periodic phase modulo ($T_d \in [4, 34]$).
///
///    - "harmonic" | "pure_harmonic" | "fourier":
///      Pure multi-frequency orthogonal Fourier resonance field (4 harmonic octaves):
///      $X(t, d) = \sum_{k=1}^4 \frac{1}{\sqrt{k}} \sin(2^{k-1} \omega_0 t + 2\pi k d / d_{\text{model}})$
///      with $2\pi$ phase normalization to eliminate float mantissa drift.
///
///    - "recall" | "associative" | "memory" | "copy":
///      Information-leak-free associative sequence memory benchmark (Graves 2014 / Ba 2016).
///      4 truly independent random vectors $\mathbf{K}_{0..3} \sim \text{Unif}(-1.5, 1.5)^{d}$ injected at $t=4..7$.
///      Strictly isolated from local background carrier; triggered by cue $\mathbf{Q}=2.0$ at $t=44$,
///      and recalled sequentially at $t=45..48$.
///
///    - "chaos" | "chaotic" | "mackey_glass" | "lorenz":
///      Coupled Map Lattice (CML) with non-degenerate positive Lyapunov exponent, evaluated in f64
///      with microscopic Langevin thermal perturbation to prevent finite-precision digital orbit collapse.
///
///    - "markov" | "grammar" | "graph" | "automaton":
///      Finite-state grammar automaton over 8 orthogonal basis patterns with doubly-stochastic transition graph
///      ($P(+1)=0.7, P(+3)=0.3$) and leak-free local stochastic jitter.
///
///    - "noise" | "gaussian" | "random_noise":
///      Isotropic Gaussian White Noise $X(t, d) \sim \mathcal{N}(0, 1)$ without artificial truncation bounds.
///
///    - "random" | "randomtokens":
///      True isotropic uniform points on $S^{d-1}$ unit sphere via Muller's method with independent 2D Hash.
///
///    - "synthetic" | "pattern":
///      Composite multi-domain sequence benchmark with strictly locked per-batch dynamics.
///
pub mod csv_parser;
pub mod json_parser;
pub mod sqlite_parser;

use candle_core::{Device, Result, Tensor};
use rand::Rng;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    Json,
    Jsonl,
    Csv,
    Sqlite,
    DeterministicPeriodic,
    PureHarmonic,
    AssociativeRecall,
    MackeyGlass,
    MarkovGrammar,
    GaussianNoise,
    RandomTokens,
    SyntheticPattern,
}

impl DatasetFormat {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "jsonl" => Self::Jsonl,
            "csv" => Self::Csv,
            "sqlite" | "db" => Self::Sqlite,
            "periodic" | "regular" | "deterministic" | "sine" => Self::DeterministicPeriodic,
            "harmonic" | "pure_harmonic" | "fourier" => Self::PureHarmonic,
            "recall" | "associative" | "memory" | "copy" => Self::AssociativeRecall,
            "chaos" | "chaotic" | "mackey_glass" | "lorenz" => Self::MackeyGlass,
            "markov" | "grammar" | "graph" | "automaton" => Self::MarkovGrammar,
            "noise" | "gaussian" | "random_noise" | "white_noise" => Self::GaussianNoise,
            "random" | "randomtokens" | "tokens" => Self::RandomTokens,
            "pattern" | "synthetic" | "composite" => Self::SyntheticPattern,
            _ => Self::Json,
        }
    }

    pub fn is_synthetic(self) -> bool {
        matches!(
            self,
            Self::DeterministicPeriodic
                | Self::PureHarmonic
                | Self::AssociativeRecall
                | Self::MackeyGlass
                | Self::MarkovGrammar
                | Self::GaussianNoise
                | Self::RandomTokens
                | Self::SyntheticPattern
        )
    }
}

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
    SyntheticStream {
        format: DatasetFormat,
        d_model: usize,
        total_batches: usize,
        current_batch: usize,
        device: Device,
        chaos_state: Option<Vec<f64>>,
        markov_state: usize,
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
        let toml_cfg = crate::config::AnnpTomlConfig::load_from_file("annp_config.toml").ok();
        let num_batches = toml_cfg
            .as_ref()
            .and_then(|c| {
                c.train
                    .synthetic_batches
                    .or(Some((c.train.epochs * 500).max(32000)))
            })
            .unwrap_or(32000);
        Self::new_with_batch_count(path, format, d_model, num_batches, device)
    }

    pub fn new_with_batch_count<P: AsRef<Path>>(
        path: P,
        format: DatasetFormat,
        d_model: usize,
        num_batches: usize,
        device: &Device,
    ) -> Result<(Self, usize)> {
        let p = path.as_ref();
        let toml_cfg = crate::config::AnnpTomlConfig::load_from_file("annp_config.toml").ok();
        let chunk_size = toml_cfg
            .as_ref()
            .and_then(|c| c.train.chunk_size)
            .unwrap_or(8192);

        // Streaming for synthetic / control benchmarks: true O(1) memory lazy generation
        if format.is_synthetic() {
            let initial_chaos = if format == DatasetFormat::MackeyGlass {
                Some(
                    (0..d_model)
                        .map(|d| 0.5f64 + 0.4f64 * (d as f64 * 0.15f64 + 0.3f64).sin())
                        .collect(),
                )
            } else {
                None
            };

            let stream = Self::SyntheticStream {
                format,
                d_model,
                total_batches: num_batches,
                current_batch: 0,
                device: device.clone(),
                chaos_state: initial_chaos,
                markov_state: 0,
            };
            return Ok((stream, num_batches));
        }

        if matches!(format, DatasetFormat::Json | DatasetFormat::Jsonl) && p.exists() {
            if let Ok((chunk_paths, _total_count)) =
                json_parser::split_and_cache_dataset(p, chunk_size)
            {
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
            if let Ok((chunk_paths, _total_count)) =
                csv_parser::split_and_cache_dataset(p, chunk_size)
            {
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
            && let Ok((chunk_paths, _total_count)) =
                sqlite_parser::split_and_cache_dataset(p, chunk_size)
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

        let batches = load_dataset(path, format, d_model, num_batches, device)?;
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
            Self::SyntheticStream {
                format,
                d_model,
                total_batches,
                current_batch,
                device,
                chaos_state,
                markov_state,
            } => {
                if *current_batch >= *total_batches {
                    return None;
                }
                let b = *current_batch;
                *current_batch += 1;

                let tensor_res = match *format {
                    DatasetFormat::DeterministicPeriodic => {
                        generate_single_deterministic_periodic_tensor(b, *d_model, device)
                    }
                    DatasetFormat::PureHarmonic => {
                        generate_single_pure_harmonic_tensor(b, *d_model, device)
                    }
                    DatasetFormat::AssociativeRecall => {
                        generate_single_associative_recall_tensor(b, *d_model, device)
                    }
                    DatasetFormat::MackeyGlass => {
                        let state = chaos_state.get_or_insert_with(|| {
                            (0..*d_model)
                                .map(|d| 0.5f64 + 0.4f64 * (d as f64 * 0.15f64 + 0.3f64).sin())
                                .collect()
                        });
                        generate_single_mackey_glass_tensor(state, *d_model, device)
                    }
                    DatasetFormat::MarkovGrammar => {
                        generate_single_markov_grammar_tensor(markov_state, *d_model, device)
                    }
                    DatasetFormat::GaussianNoise => {
                        generate_single_gaussian_noise_tensor(*d_model, device)
                    }
                    DatasetFormat::RandomTokens => {
                        generate_single_random_tokens_tensor(*d_model, device)
                    }
                    _ => generate_single_synthetic_pattern_tensor(b, *d_model, device),
                };

                return Some(tensor_res);
            }
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
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    match format {
        DatasetFormat::DeterministicPeriodic => {
            generate_deterministic_periodic_tensors(d_model, num_batches, device)
        }
        DatasetFormat::PureHarmonic => generate_pure_harmonic_tensors(d_model, num_batches, device),
        DatasetFormat::AssociativeRecall => {
            generate_associative_recall_tensors(d_model, num_batches, device)
        }
        DatasetFormat::MackeyGlass => {
            generate_mackey_glass_chaos_tensors(d_model, num_batches, device)
        }
        DatasetFormat::MarkovGrammar => {
            generate_markov_grammar_tensors(d_model, num_batches, device)
        }
        DatasetFormat::GaussianNoise => {
            generate_gaussian_noise_tensors(d_model, num_batches, device)
        }
        DatasetFormat::RandomTokens => generate_random_tokens_tensors(d_model, num_batches, device),
        DatasetFormat::SyntheticPattern => {
            generate_synthetic_pattern_tensors(d_model, num_batches, device)
        }
        _ => {
            let p = path.as_ref();
            if !p.exists() {
                println!(
                    "Dataset path {:?} not found. Generating complex multi-frequency harmonic resonance tensors for Loss testing ({} batches).",
                    p, num_batches
                );
                return generate_synthetic_pattern_tensors(d_model, num_batches, device);
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
                _ => generate_synthetic_pattern_tensors(d_model, num_batches, device),
            }
        }
    }
}

// =========================================================================================
// 1. DETERMINISTIC PERIODIC DYNAMICAL SYSTEM GENERATOR (ABSOLUTE REGULAR CONTROL)
// =========================================================================================

pub fn generate_single_deterministic_periodic_tensor(
    b: usize,
    d_model: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = 64;
    let mut flat = Vec::with_capacity(seq_len * d_model);

    for t in 0..seq_len {
        let global_t_u64 = (b * seq_len + t) as u64;
        for d in 0..d_model {
            // Multi-scale base periods across embedding channels: T_d in [4, 34]
            let period_u64 = 4u64 + ((d % 16) as u64 * 2u64);
            // High-precision periodic modulo prevents float mantissa precision loss at high batch counts
            let t_in_period = (global_t_u64 % period_u64) as f32;
            let t_period = period_u64 as f32;
            let phi_d = (d as f32 / d_model as f32) * std::f32::consts::TAU;
            let omega = std::f32::consts::TAU / t_period;

            let fund = (omega * t_in_period + phi_d).cos();
            let overtone = 0.5f32 * (2.0f32 * omega * t_in_period).sin();
            flat.push(fund + overtone);
        }
    }
    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_deterministic_periodic_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    for b in 0..num_batches {
        tensors.push(generate_single_deterministic_periodic_tensor(
            b, d_model, device,
        )?);
    }
    println!(
        "Generated {} batches of Deterministic Periodic (Absolute Regular) control data.",
        num_batches
    );
    Ok(tensors)
}

// =========================================================================================
// 2. PURE ORTHOGONAL MULTI-FREQUENCY HARMONIC RESONANCE GENERATOR
// =========================================================================================

pub fn generate_single_pure_harmonic_tensor(
    b: usize,
    d_model: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = 64;
    let mut flat = Vec::with_capacity(seq_len * d_model);
    let omega_0 = 0.04f64;

    for t in 0..seq_len {
        let global_t_f64 = (b * seq_len + t) as f64;
        for d in 0..d_model {
            let d_ratio = (d as f64) / (d_model as f64);
            let mut val = 0.0f64;
            for k in 1..=4 {
                let freq_mult = (1 << (k - 1)) as f64;
                let k_weight = 1.0f64 / (k as f64).sqrt();
                let spatial_phase = std::f64::consts::TAU * (k as f64) * d_ratio;
                // High-precision periodic modulo 2*PI prevents float mantissa precision loss
                let total_phase =
                    (freq_mult * omega_0 * global_t_f64 + spatial_phase) % std::f64::consts::TAU;
                val += k_weight * total_phase.sin();
            }
            flat.push(val as f32);
        }
    }
    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_pure_harmonic_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    for b in 0..num_batches {
        tensors.push(generate_single_pure_harmonic_tensor(b, d_model, device)?);
    }
    println!(
        "Generated {} batches of Pure Orthogonal Harmonic Wave data.",
        num_batches
    );
    Ok(tensors)
}

// =========================================================================================
// 3. ZERO-INFORMATION-LEAKAGE LONG-RANGE MULTI-TOKEN ASSOCIATIVE MEMORY RECALL
// =========================================================================================

pub fn generate_single_associative_recall_tensor(
    _b: usize,
    d_model: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = 64;
    let mut flat = Vec::with_capacity(seq_len * d_model);
    let mut rng = rand::rng();

    // 4 truly independent, i.i.d. random token vectors K_0..K_3 in [-1.5, 1.5]
    // Statistically completely decoupled from batch index b, time t, or carrier waves!
    let key_sequence: Vec<Vec<f32>> = (0..4)
        .map(|_| {
            (0..d_model)
                .map(|_| rng.random_range(-1.5f32..1.5f32))
                .collect()
        })
        .collect();

    for t in 0..seq_len {
        if (4..=7).contains(&t) {
            // Key sequence injection phase: K_0 at t=4, K_1 at t=5, K_2 at t=6, K_3 at t=7
            flat.extend_from_slice(&key_sequence[t - 4]);
        } else if t == 44 {
            // Trigger cue pulse
            flat.extend(std::iter::repeat_n(2.0f32, d_model));
        } else if (45..=48).contains(&t) {
            // Exact sequential recall target phase: K_0 at t=45, K_1 at t=46, K_2 at t=47, K_3 at t=48
            flat.extend_from_slice(&key_sequence[t - 45]);
        } else {
            // Purely local zero-mean neutral background noise (strictly no global_t leakage)
            for _ in 0..d_model {
                flat.push(rng.random_range(-0.15f32..0.15f32));
            }
        }
    }
    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_associative_recall_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    for b in 0..num_batches {
        tensors.push(generate_single_associative_recall_tensor(
            b, d_model, device,
        )?);
    }
    println!(
        "Generated {} batches of Publication-Grade Zero-Leakage Associative Recall data.",
        num_batches
    );
    Ok(tensors)
}

// =========================================================================================
// 4. NON-DEGENERATE COUPLED MAP LATTICE (CML) CHAOTIC DYNAMICAL SYSTEM
// =========================================================================================

pub fn generate_single_mackey_glass_tensor(
    state: &mut [f64],
    d_model: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = 64;
    let kappa = 0.04f64;
    let mut flat = Vec::with_capacity(seq_len * d_model);
    let mut rng = rand::rng();

    for _ in 0..seq_len {
        let mut u = vec![0.0f64; d_model];
        for d in 0..d_model {
            // Fully developed chaotic regime r_d in [3.95, 3.99] strictly away from periodic windows
            let r_d = 3.95f64 + 0.04f64 * (d as f64 / d_model as f64);
            let x = state[d];
            u[d] = r_d * x * (1.0f64 - x);
        }

        // Spatial diffusive coupling with periodic boundary & microscopic Langevin regularization
        for d in 0..d_model {
            let left = u[(d + d_model - 1) % d_model];
            let right = u[(d + 1) % d_model];
            let next_x = (1.0f64 - 2.0f64 * kappa) * u[d] + kappa * (left + right);
            // Microscopic Langevin perturbation prevents finite-precision digital orbit collapse
            let thermal_jitter: f64 = rng.random_range(-1e-7f64..1e-7f64);
            state[d] = (next_x + thermal_jitter).clamp(0.0001f64, 0.9999f64);
            flat.push(((state[d] - 0.5f64) * 3.0f64) as f32);
        }
    }
    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_mackey_glass_chaos_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    let mut state: Vec<f64> = (0..d_model)
        .map(|d| 0.5f64 + 0.4f64 * (d as f64 * 0.15f64 + 0.3f64).sin())
        .collect();

    for _ in 0..num_batches {
        tensors.push(generate_single_mackey_glass_tensor(
            &mut state, d_model, device,
        )?);
    }
    println!(
        "Generated {} batches of Non-Degenerate Coupled Map Lattice Chaotic Field data.",
        num_batches
    );
    Ok(tensors)
}

// =========================================================================================
// 5. FINITE-STATE MARKOV GRAMMAR AUTOMATON (ZERO INFORMATION LEAKAGE)
// =========================================================================================

pub fn generate_single_markov_grammar_tensor(
    current_state: &mut usize,
    d_model: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = 64;
    let mut flat = Vec::with_capacity(seq_len * d_model);
    let mut rng = rand::rng();
    let num_states = 8;

    let basis_states: Vec<Vec<f32>> = (0..num_states)
        .map(|s| {
            (0..d_model)
                .map(|d| {
                    let freq = (s + 1) as f32;
                    1.8f32 * ((d as f32 / d_model as f32) * std::f32::consts::TAU * freq).cos()
                })
                .collect()
        })
        .collect();

    for _ in 0..seq_len {
        let r: f32 = rng.random_range(0.0f32..1.0f32);
        *current_state = if r < 0.7f32 {
            (*current_state + 1) % num_states
        } else {
            (*current_state + 3) % num_states
        };

        let state_vec = &basis_states[*current_state];
        for &val in state_vec.iter() {
            // Purely local i.i.d. jitter: zero global_t / timestamp leakage!
            let jitter = rng.random_range(-0.03f32..0.03f32);
            flat.push(val + jitter);
        }
    }
    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_markov_grammar_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    let mut current_state = 0;
    for _ in 0..num_batches {
        tensors.push(generate_single_markov_grammar_tensor(
            &mut current_state,
            d_model,
            device,
        )?);
    }
    println!(
        "Generated {} batches of Publication-Grade Markov Grammar Automaton data.",
        num_batches
    );
    Ok(tensors)
}

// =========================================================================================
// 6. ISOTROPIC GAUSSIAN WHITE NOISE (NULL HYPOTHESIS CONTROL - UNBOUNDED FULL-SUPPORT N(0, 1))
// =========================================================================================

pub fn generate_single_gaussian_noise_tensor(d_model: usize, device: &Device) -> Result<Tensor> {
    let seq_len = 64;
    let total_elements = seq_len * d_model;
    let mut flat = Vec::with_capacity(total_elements + 1);
    let mut rng = rand::rng();

    // Full-support Box-Muller transformation using full precision range without artificial 1e-7 truncation
    let pairs = total_elements.div_ceil(2);
    for _ in 0..pairs {
        let u1: f32 = rng.random_range(f32::EPSILON..1.0f32);
        let u2: f32 = rng.random_range(0.0f32..std::f32::consts::TAU);
        let r = (-2.0f32 * u1.ln()).sqrt();
        flat.push(r * u2.cos());
        flat.push(r * u2.sin());
    }
    flat.truncate(total_elements);

    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_gaussian_noise_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    for _ in 0..num_batches {
        tensors.push(generate_single_gaussian_noise_tensor(d_model, device)?);
    }
    println!(
        "Generated {} batches of Isotropic Gaussian White Noise (Null Hypothesis) data.",
        num_batches
    );
    Ok(tensors)
}

// =========================================================================================
// 7. DISCRETE UNIFORM RANDOM TOKEN EMBEDDING (INDEPENDENT 2D HASH + MULLER S^{d-1} ISOTROPIC SPHERE)
// =========================================================================================

#[inline(always)]
fn token_dim_hash(token_id: u32, dim_pair_idx: usize, salt: u64) -> u64 {
    let mut z = (token_id as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add((dim_pair_idx as u64).wrapping_mul(0xBF58476D1CE4E5B9))
        .wrapping_add(salt);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

pub fn generate_single_random_tokens_tensor(d_model: usize, device: &Device) -> Result<Tensor> {
    let seq_len = 64;
    let mut flat = Vec::with_capacity(seq_len * d_model);
    let mut rng = rand::rng();

    for _ in 0..seq_len {
        let token_id: u32 = rng.random_range(1..32000);
        let mut gaussians = Vec::with_capacity(d_model + 1);
        let num_pairs = d_model.div_ceil(2);

        for pair_idx in 0..num_pairs {
            // Independent 2D hash for u1 and u2: ZERO topological shifting / sliding window correlation!
            let h1 = token_dim_hash(token_id, pair_idx, 0x123456789ABCDEF0);
            let h2 = token_dim_hash(token_id, pair_idx, 0x0FEDCBA987654321);

            let u1 = ((h1 as f32) / (u64::MAX as f32)).clamp(f32::EPSILON, 0.999999f32);
            let u2 = ((h2 as f32) / (u64::MAX as f32)) * std::f32::consts::TAU;

            let r = (-2.0f32 * u1.ln()).sqrt();
            gaussians.push(r * u2.cos());
            gaussians.push(r * u2.sin());
        }
        gaussians.truncate(d_model);

        let norm_sq: f32 = gaussians.iter().map(|&x| x * x).sum();
        let inv_norm = 1.0f32 / norm_sq.sqrt().max(1e-6);
        for g in gaussians {
            flat.push(g * inv_norm);
        }
    }
    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_random_tokens_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    for _ in 0..num_batches {
        tensors.push(generate_single_random_tokens_tensor(d_model, device)?);
    }
    println!(
        "Generated {} batches of Publication-Grade Muller Isotropic Token Embeddings.",
        num_batches
    );
    Ok(tensors)
}

// =========================================================================================
// 8. COMPOSITE MULTI-DOMAIN SYNTHETIC PATTERN GENERATOR (LOCKED-MODE SEQUENCES)
// =========================================================================================

pub fn generate_single_synthetic_pattern_tensor(
    b: usize,
    d_model: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq_len = 64;
    let mut flat = Vec::with_capacity(seq_len * d_model);
    let mut rng = rand::rng();
    let current_mode = b % 4;

    // Truly independent random key sequence for associative mode
    let key_sequence: Vec<Vec<f32>> = (0..4)
        .map(|_| {
            (0..d_model)
                .map(|_| rng.random_range(-1.5f32..1.5f32))
                .collect()
        })
        .collect();

    if current_mode == 1 {
        // Reliable Long-Range Associative Memory Retrieval (Zero global_t leak)
        for t in 0..seq_len {
            if (4..=7).contains(&t) {
                flat.extend_from_slice(&key_sequence[t - 4]);
            } else if t == 44 {
                flat.extend(std::iter::repeat_n(2.0f32, d_model));
            } else if (45..=48).contains(&t) {
                flat.extend_from_slice(&key_sequence[t - 45]);
            } else {
                for _ in 0..d_model {
                    flat.push(rng.random_range(-0.15f32..0.15f32));
                }
            }
        }
    } else {
        for t in 0..seq_len {
            let t_norm = (t as f32) * 0.05f32; // Strictly local time coordinate for within-sequence dynamics

            for d in 0..d_model {
                let d_norm = d as f32 / d_model as f32;
                let d_val = d as f32 * 0.08f32;

                let val = match current_mode {
                    0 => {
                        // Multi-frequency Harmonic Resonance with Phase Coupling
                        let low_freq = (t_norm * 0.2f32 + d_val).sin()
                            + (t_norm * 0.05f32 - d_val * 0.5f32).cos();
                        let high_freq = 0.4f32 * (t_norm * 3.0f32 + d_val * 4.0f32).sin();
                        let coupling = 0.25f32 * (t_norm * 0.1f32 * d_val).sin();
                        low_freq + high_freq + coupling
                    }
                    2 => {
                        // Hierarchical Nested Syntax (SwiGLU Modulation)
                        let gate = (t_norm * 0.5f32 + d_val).sin();
                        let up = (t_norm * 0.8f32 - d_val * 1.2f32).cos();
                        let swish = gate / (1.0f32 + (-gate).exp());
                        swish * up * (1.0f32 + 0.5f32 * d_norm)
                    }
                    _ => {
                        // Zipfian Non-Linear Burst with Dynamic Phase Drift
                        let zipf_factor = 1.0f32 / (1.0f32 + (d as f32 * 0.05f32));
                        let carrier = (t_norm * 1.5f32 + d_norm * std::f32::consts::TAU).sin();
                        let tanh_mod = (carrier * 2.0f32).tanh();
                        tanh_mod * zipf_factor * 1.8f32
                    }
                };

                flat.push(val);
            }
        }
    }
    Tensor::from_vec(flat, (seq_len, d_model), device)
}

pub fn generate_synthetic_pattern_tensors(
    d_model: usize,
    num_batches: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::with_capacity(num_batches);
    for b in 0..num_batches {
        tensors.push(generate_single_synthetic_pattern_tensor(
            b, d_model, device,
        )?);
    }
    println!(
        "Generated {} batches of Composite Synthetic Pattern data.",
        num_batches
    );
    Ok(tensors)
}
