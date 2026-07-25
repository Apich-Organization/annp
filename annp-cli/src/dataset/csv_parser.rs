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

    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut float_row = Vec::new();
        for field in record.iter() {
            if let Ok(val) = field.trim().parse::<f32>() {
                float_row.push(val);
            }
        }
        if !float_row.is_empty() {
            rows.push(float_row);
        }
    }

    let mut tensors = Vec::new();
    let chunk_size = 8; // Group 8 sequence rows per sequence tensor

    for chunk in rows.chunks(chunk_size) {
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
