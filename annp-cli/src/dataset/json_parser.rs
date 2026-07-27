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
            let t = Tensor::from_vec(flat, (seq_len, d_model), device)?;
            return Ok(Some(t));
        }
    }
    Ok(None)
}
