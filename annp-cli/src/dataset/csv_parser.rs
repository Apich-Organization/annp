use crate::tokenizer::AnnpTokenizer;
use candle_core::{Device, Result, Tensor};
use std::path::Path;

/// Parses CSV format dataset files into sequence tensors [seq_len, d_model]
pub fn load_csv_dataset<P: AsRef<Path>>(
    path: P,
    d_model: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path.as_ref())
        .map_err(|e| candle_core::Error::Msg(format!("CSV Open error: {}", e)))?;

    let tokenizer = AnnpTokenizer::load_from_file("tokenizer.model");
    let mut tensors = Vec::new();
    let mut float_rows = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut row_text = String::new();
        let mut row_floats = Vec::new();

        for field in record.iter() {
            let trimmed = field.trim();
            if let Ok(val) = trimmed.parse::<f32>() {
                row_floats.push(val);
            } else if !trimmed.is_empty() {
                if !row_text.is_empty() {
                    row_text.push(' ');
                }
                row_text.push_str(trimmed);
            }
        }

        if !row_text.is_empty() {
            if let Ok(t) = tokenizer.encode_to_tensor(&row_text, d_model, device) {
                tensors.push(t);
            }
        } else if !row_floats.is_empty() {
            float_rows.push(row_floats);
        }
    }

    let chunk_size = 8;
    for chunk in float_rows.chunks(chunk_size) {
        let seq_len = chunk.len();
        let mut flat = Vec::with_capacity(seq_len * d_model);

        for row in chunk {
            if row.len() >= d_model {
                flat.extend_from_slice(&row[..d_model]);
            } else {
                flat.extend_from_slice(row);
                flat.resize(flat.len() + (d_model - row.len()), 0.0);
            }
        }

        if flat.len() == seq_len * d_model {
            let t = Tensor::from_vec(flat, (seq_len, d_model), device)?;
            tensors.push(t);
        }
    }

    if tensors.is_empty() {
        tensors.push(Tensor::randn(0.0f32, 1.0f32, (8, d_model), device)?);
    }

    Ok(tensors)
}
