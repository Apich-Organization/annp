use candle_core::{Device, Result as CandleResult, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

use std::collections::HashMap;
use std::fs;

pub struct AnnpTokenizer {
    inner: Option<Tokenizer>,
    spm_vocab: Option<(HashMap<String, u32>, Vec<String>)>,
}

fn parse_spm_vocab(buf: &[u8]) -> Option<(HashMap<String, u32>, Vec<String>)> {
    // Basic SentencePiece binary header identification (0x08, 0x0A)
    if buf.len() < 100 {
        return None;
    }
    let mut vocab = HashMap::new();
    let mut inv_vocab = Vec::new();

    // Simple naive parser for SentencePiece protobuf-like structure
    // This looks for string pieces embedded in the model binary
    for i in 0..(buf.len() - 32) {
        if buf[i] == 0x0A && buf[i + 1] < 32 {
            let len = buf[i + 1] as usize;
            let slice = &buf[i + 2..i + 2 + len];
            if let Ok(s) = std::str::from_utf8(slice) {
                if s.len() > 0
                    && s.chars()
                        .all(|c| c.is_alphanumeric() || c == '_' || c == '\u{2581}')
                {
                    if !vocab.contains_key(s) {
                        let id = inv_vocab.len() as u32;
                        vocab.insert(s.to_string(), id);
                        inv_vocab.push(s.to_string());
                    }
                }
            }
        }
    }
    if inv_vocab.is_empty() {
        None
    } else {
        Some((vocab, inv_vocab))
    }
}

impl AnnpTokenizer {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Self {
        let p = path.as_ref();

        // 1. Try HF Tokenizer JSON directly
        if p.exists() {
            if let Ok(t) = Tokenizer::from_file(p) {
                println!(
                    "Successfully loaded Hugging Face JSON Tokenizer from: {:?}",
                    p
                );
                return Self {
                    inner: Some(t),
                    spm_vocab: None,
                };
            }
        }

        // 2. Try tokenizer.json if specified file is tokenizer.model
        let json_path = p.with_extension("json");
        if json_path.exists() {
            if let Ok(t) = Tokenizer::from_file(&json_path) {
                println!(
                    "Successfully loaded Hugging Face JSON Tokenizer from: {:?}",
                    json_path
                );
                return Self {
                    inner: Some(t),
                    spm_vocab: None,
                };
            }
        }

        // 3. Try Native SentencePiece Protobuf binary parser
        if p.exists() {
            if let Ok(buf) = fs::read(p) {
                if let Some((vocab, inv_vocab)) = parse_spm_vocab(&buf) {
                    println!(
                        "Successfully loaded SentencePiece Binary Model from: {:?} (Vocabulary Size: {} tokens)",
                        p,
                        inv_vocab.len()
                    );
                    return Self {
                        inner: None,
                        spm_vocab: Some((vocab, inv_vocab)),
                    };
                }
            }
        }

        println!(
            "Notice: Tokenizer file {:?} unavailable or unrecognized. Initializing Byte-Level ASCII/UTF8 fallback tokenizer.",
            p
        );
        Self {
            inner: None,
            spm_vocab: None,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(ref t) = self.inner
            && let Ok(encoding) = t.encode(text, true)
        {
            return encoding.get_ids().to_vec();
        }

        if let Some((ref vocab, _)) = self.spm_vocab {
            let mut ids = Vec::new();
            let spm_text = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
            let chars: Vec<char> = spm_text.chars().collect();
            let mut i = 0;

            while i < chars.len() {
                let mut matched = false;
                // Try longest prefix match
                for len in (1..=(chars.len() - i).min(16)).rev() {
                    let sub: String = chars[i..i + len].iter().collect();
                    if let Some(&id) = vocab.get(&sub) {
                        ids.push(id);
                        i += len;
                        matched = true;
                        break;
                    }
                }

                if !matched {
                    let ch = chars[i];
                    let ch_str = ch.to_string();
                    if let Some(&id) = vocab.get(&ch_str) {
                        ids.push(id);
                    } else {
                        // Byte fallback
                        for b in ch_str.as_bytes() {
                            ids.push(*b as u32);
                        }
                    }
                    i += 1;
                }
            }

            if !ids.is_empty() {
                return ids;
            }
        }

        text.bytes().map(|b| b as u32).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(ref t) = self.inner
            && let Ok(text) = t.decode(ids, true)
        {
            return text;
        }

        if let Some((_, ref inv_vocab)) = self.spm_vocab {
            let mut res = String::new();
            for &id in ids {
                let idx = id as usize;
                if idx < inv_vocab.len() {
                    res.push_str(&inv_vocab[idx]);
                } else {
                    res.push(char::from_u32(id).unwrap_or('?'));
                }
            }
            return res.replace('\u{2581}', " ");
        }

        let bytes: Vec<u8> = ids.iter().map(|&id| (id & 0xFF) as u8).collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Convert tokenized sequence into dense ANNP Input Activation Tensor [seq_len, d_model]
    pub fn encode_to_tensor(
        &self,
        text: &str,
        d_model: usize,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let ids = self.encode(text);
        let seq_len = ids.len().max(1);
        let mut flat = Vec::with_capacity(seq_len * d_model);

        for (pos, &token_id) in ids.iter().enumerate() {
            let base_val = (token_id as f32) / 1000.0f32;
            let mut tok_vec = Vec::with_capacity(d_model);
            for d in 0..d_model {
                let phase = (pos as f32 * 0.1f32 + d as f32 * 0.05f32).sin();
                let freq = (base_val + d as f32 * 0.01f32).cos();
                tok_vec.push(base_val * phase + freq * 0.5f32);
            }
            // RMS Normalization to keep embedding magnitude at unit scale (~1.0)
            let rms = (tok_vec.iter().map(|v| v * v).sum::<f32>() / d_model as f32)
                .sqrt()
                .max(1e-6);
            for v in tok_vec {
                flat.push(v / rms);
            }
        }

        Tensor::from_vec(flat, (seq_len, d_model), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokenizer = AnnpTokenizer::load_from_file("tokenizer.model");
        let text = "ANNP Asynchronous Neural Network Protocol Simulation";
        let ids = tokenizer.encode(text);
        assert!(!ids.is_empty());
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.is_empty());
    }
}
