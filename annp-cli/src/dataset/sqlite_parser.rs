use crate::tokenizer::AnnpTokenizer;
use candle_core::{Device, Result, Tensor};
use rusqlite::Connection;
use std::path::Path;

/// Parses SQLite database tables containing text or vector embeddings into sequence tensors [seq_len, d_model]
pub fn load_sqlite_dataset<P: AsRef<Path>>(
    path: P,
    d_model: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let conn = Connection::open(path.as_ref())
        .map_err(|e| candle_core::Error::Msg(format!("SQLite Connection error: {}", e)))?;

    let tokenizer = AnnpTokenizer::load_from_file("tokenizer.model");
    let mut tensors = Vec::new();

    // Check if table contains text column first
    if let Ok(mut stmt) = conn
        .prepare("SELECT content FROM dataset LIMIT 100")
        .or_else(|_| conn.prepare("SELECT text FROM dataset LIMIT 100"))
        .or_else(|_| conn.prepare("SELECT content FROM samples LIMIT 100"))
        && let Ok(rows) = stmt.query_map([], |row| {
            let txt: String = row.get(0)?;
            Ok(txt)
        })
    {
        for row in rows.flatten() {
            if let Ok(t) = tokenizer.encode_to_tensor(&row, d_model, device) {
                tensors.push(t);
            }
        }
    }

    if tensors.is_empty()
        && let Ok(mut stmt) = conn
            .prepare("SELECT payload FROM embeddings LIMIT 100")
            .or_else(|_| conn.prepare("SELECT data FROM dataset LIMIT 100"))
    {
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(blob)
            })
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let mut current_seq = Vec::new();
        let seq_len = 8;

        for row in rows.flatten() {
            let floats: Vec<f32> = row
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            if floats.len() >= d_model {
                current_seq.extend_from_slice(&floats[..d_model]);
            } else {
                let mut padded = floats;
                padded.resize(d_model, 0.0);
                current_seq.extend_from_slice(&padded);
            }

            if current_seq.len() == seq_len * d_model {
                let t =
                    Tensor::from_vec(std::mem::take(&mut current_seq), (seq_len, d_model), device)?;
                tensors.push(t);
            }
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
    use std::io::Write;
    use std::path::PathBuf;

    let tmp_dir = PathBuf::from("tmp").join("annp_chunks");
    let _ = fs::create_dir_all(&tmp_dir);

    let p = path.as_ref();
    let conn = Connection::open(p)
        .map_err(|e| candle_core::Error::Msg(format!("SQLite Connection error: {}", e)))?;

    let mut chunk_paths = Vec::new();
    let mut current_chunk = Vec::new();
    let mut chunk_idx = 0;
    let mut total_count = 0;

    let mut flush_chunk = |records: &mut Vec<String>| -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let chunk_path = tmp_dir.join(format!(
            "{}_chunk_{}.txt",
            p.file_name().unwrap().to_string_lossy(),
            chunk_idx
        ));
        let mut file = fs::File::create(&chunk_path).unwrap();
        for rec in records.iter() {
            writeln!(file, "{}", rec).unwrap();
        }
        chunk_paths.push(chunk_path);
        chunk_idx += 1;
        records.clear();
        Ok(())
    };

    if let Ok(mut stmt) = conn
        .prepare("SELECT content FROM dataset")
        .or_else(|_| conn.prepare("SELECT text FROM dataset"))
        .or_else(|_| conn.prepare("SELECT content FROM samples"))
    {
        let mut rows = stmt.query([]).unwrap();
        while let Ok(Some(row)) = rows.next() {
            if let Ok(txt) = row.get::<_, String>(0) {
                current_chunk.push(txt);
                total_count += 1;
                if current_chunk.len() >= chunk_size {
                    flush_chunk(&mut current_chunk)?;
                }
            }
        }
    }
    flush_chunk(&mut current_chunk)?;

    Ok((chunk_paths, total_count))
}

pub fn load_sqlite_chunk<P: AsRef<Path>>(
    path: P,
    d_model: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let tokenizer = AnnpTokenizer::load_from_file("tokenizer.model");
    let mut tensors = Vec::new();
    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(l) = line {
                if let Ok(t) = tokenizer.encode_to_tensor(&l, d_model, device) {
                    tensors.push(t);
                }
            }
        }
    }
    Ok(tensors)
}
