use crate::tokenizer::AnnpTokenizer;
use candle_core::{Device, Result, Tensor};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parses JSON or JSONL format files into sequence tensors [seq_len, d_model]
pub fn load_json_or_jsonl_dataset<P: AsRef<Path>>(
    path: P,
    is_jsonl: bool,
    d_model: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let file = File::open(path.as_ref())
        .map_err(|e| candle_core::Error::Msg(format!("Failed to open JSON dataset: {}", e)))?;
    let reader = BufReader::new(file);
    let mut tensors = Vec::new();
    let tokenizer = AnnpTokenizer::load_from_file("tokenizer.model");

    if is_jsonl {
        for line in reader.lines() {
            let l = line.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            if l.trim().is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(&l)
                && let Some(t) = parse_value_to_tensor(&v, &tokenizer, d_model, device)?
            {
                tensors.push(t);
            }
        }
    } else {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read JSON dataset: {}", e)))?;

        match serde_json::from_str::<Value>(&content) {
            Ok(v) => {
                if let Some(arr) = v.as_array() {
                    for item in arr {
                        if let Some(t) = parse_value_to_tensor(item, &tokenizer, d_model, device)? {
                            tensors.push(t);
                        }
                    }
                } else if let Some(t) = parse_value_to_tensor(&v, &tokenizer, d_model, device)? {
                    tensors.push(t);
                }
            }
            Err(_) => {
                // Auto-fallback: file is formatted as JSONL (line-delimited JSON objects)
                for line in content.lines() {
                    let l = line.trim();
                    if l.is_empty() {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<Value>(l)
                        && let Some(t) = parse_value_to_tensor(&v, &tokenizer, d_model, device)?
                    {
                        tensors.push(t);
                    }
                }
            }
        }
    }

    if tensors.is_empty() {
        tensors.push(Tensor::randn(0.0f32, 1.0f32, (8, d_model), device)?);
    }

    Ok(tensors)
}

pub fn parse_value_to_tensor(
    val: &Value,
    tokenizer: &AnnpTokenizer,
    d_model: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    if let Some(text) = val
        .get("input_text")
        .or_else(|| val.get("text"))
        .or_else(|| val.get("content"))
        .and_then(|v| v.as_str())
    {
        let t = tokenizer.encode_to_tensor(text, d_model, device)?;
        return Ok(Some(t));
    }

    if let Some(array) = val
        .get("embeddings")
        .or_else(|| val.get("tokens"))
        .and_then(|v| v.as_array())
    {
        if let Some(first) = array.first() {
            if !first.is_array() {
                let ids: Vec<u32> = array
                    .iter()
                    .filter_map(|v| v.as_i64().map(|id| id as u32))
                    .collect();
                if !ids.is_empty() {
                    let t = tokenizer.encode_ids_to_tensor(&ids, d_model, device)?;
                    return Ok(Some(t));
                }
            }
        }

        let mut flat = Vec::new();
        let mut seq_len = 0;

        for item in array {
            if let Some(vec) = item.as_array() {
                seq_len += 1;
                for f in vec {
                    flat.push(f.as_f64().unwrap_or(0.0) as f32);
                }
            }
        }

        if seq_len > 0 && flat.len() == seq_len * d_model {
            let rms = (flat.iter().map(|x| x * x).sum::<f32>() / flat.len() as f32)
                .sqrt()
                .max(1e-6);
            for x in flat.iter_mut() {
                *x /= rms;
            }
            let t = Tensor::from_vec(flat, (seq_len, d_model), device)?;
            return Ok(Some(t));
        }
    }
    Ok(None)
}

pub fn split_and_cache_dataset<P: AsRef<Path>>(
    path: P,
    chunk_size: usize,
) -> Result<(Vec<std::path::PathBuf>, usize)> {
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;

    let tmp_dir = PathBuf::from("tmp").join("annp_chunks");
    let _ = fs::create_dir_all(&tmp_dir);

    let p = path.as_ref();
    let mut all_values = Vec::new();

    // 1. Try reading whole JSON file (handles multiline JSON array, pretty JSON, or single object)
    if let Ok(file) = File::open(p)
        && let Ok(v) = serde_json::from_reader::<_, Value>(BufReader::new(file))
    {
        if let Some(arr) = v.as_array() {
            all_values = arr.clone();
        } else {
            all_values.push(v);
        }
    }

    // 2. Fallback: line-by-line JSONL format
    if all_values.is_empty()
        && let Ok(file) = File::open(p)
    {
        let reader = BufReader::new(file);
        for l in reader.lines().map_while(|l| l.ok()) {
            let trimmed = l.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
                if let Some(arr) = v.as_array() {
                    all_values.extend(arr.clone());
                } else {
                    all_values.push(v);
                }
            }
        }
    }

    if all_values.is_empty() {
        return Err(candle_core::Error::Msg(format!(
            "Dataset {:?} contains 0 valid JSON items",
            p
        )));
    }

    println!(
        "[DATASET] Converting JSON to JSONL in tmp: Loaded {} total valid samples from {:?}",
        all_values.len(),
        p.file_name().unwrap_or_default()
    );

    // 3. Save as chunk files in tmp/annp_chunks/ (e.g. chunk_0.jsonl, chunk_1.jsonl...)
    let mut chunk_paths = Vec::new();

    for (chunk_idx, chunk_values) in all_values.chunks(chunk_size).enumerate() {
        let chunk_file_path = tmp_dir.join(format!("chunk_{:05}.jsonl", chunk_idx));
        let mut out_file = File::create(&chunk_file_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create tmp chunk: {}", e)))?;

        for val in chunk_values {
            if let Ok(json_line) = serde_json::to_string(val) {
                let _ = writeln!(out_file, "{}", json_line);
            }
        }

        chunk_paths.push(chunk_file_path);
    }

    let total_samples = all_values.len();
    Ok((chunk_paths, total_samples))
}
