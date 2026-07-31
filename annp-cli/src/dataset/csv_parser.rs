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

pub fn split_and_cache_dataset<P: AsRef<Path>>(
    path: P,
    chunk_size: usize,
) -> Result<(Vec<std::path::PathBuf>, usize)> {
    use std::fs;
    use std::path::PathBuf;

    let tmp_dir = PathBuf::from("tmp").join("annp_chunks");
    let _ = fs::create_dir_all(&tmp_dir);

    let p = path.as_ref();
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(p)
        .map_err(|e| candle_core::Error::Msg(format!("CSV Open error: {}", e)))?;

    let headers = reader.headers().unwrap().clone();

    let mut chunk_paths = Vec::new();
    let mut current_chunk = Vec::new();
    let mut chunk_idx = 0;
    let mut total_count = 0;

    let mut flush_chunk = |records: &mut Vec<csv::StringRecord>| -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let chunk_path = tmp_dir.join(format!(
            "{}_chunk_{}.csv",
            p.file_name().unwrap().to_string_lossy(),
            chunk_idx
        ));
        let mut wtr = csv::Writer::from_path(&chunk_path).unwrap();
        wtr.write_record(&headers).unwrap();
        for rec in records.iter() {
            wtr.write_record(rec).unwrap();
        }
        wtr.flush().unwrap();
        chunk_paths.push(chunk_path);
        chunk_idx += 1;
        records.clear();
        Ok(())
    };

    for result in reader.records() {
        if let Ok(record) = result {
            current_chunk.push(record);
            total_count += 1;
            if current_chunk.len() >= chunk_size {
                flush_chunk(&mut current_chunk)?;
            }
        }
    }
    flush_chunk(&mut current_chunk)?;

    Ok((chunk_paths, total_count))
}
