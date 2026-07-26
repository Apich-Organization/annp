use candle_core::{Device, Result as CandleResult, Tensor};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct AnnpTokenizer {
    inner: Option<Tokenizer>,
}

impl AnnpTokenizer {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Self {
        let p = path.as_ref();
        if p.exists() {
            match Tokenizer::from_file(p) {
                Ok(t) => {
                    println!("Successfully loaded Hugging Face Tokenizer from: {:?}", p);
                    Self { inner: Some(t) }
                }
                Err(e) => {
                    println!(
                        "Note: Loading HF JSON tokenizer from {:?} returned: {}. Initializing Tokenizer wrapper with fallback.",
                        p, e
                    );
                    Self { inner: None }
                }
            }
        } else {
            println!(
                "Tokenizer file {:?} not found. Falling back to built-in byte/ASCII tokenizer.",
                p
            );
            Self { inner: None }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(ref t) = self.inner {
            if let Ok(encoding) = t.encode(text, true) {
                return encoding.get_ids().to_vec();
            }
        }
        text.bytes().map(|b| b as u32).collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(ref t) = self.inner {
            if let Ok(text) = t.decode(ids, true) {
                return text;
            }
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
            for d in 0..d_model {
                let phase = (pos as f32 * 0.1f32 + d as f32 * 0.05f32).sin();
                let freq = (base_val + d as f32 * 0.01f32).cos();
                flat.push(base_val * phase + freq * 0.5f32);
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
