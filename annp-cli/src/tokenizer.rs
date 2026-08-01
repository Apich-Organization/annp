use candle_core::{Device, Result as CandleResult, Tensor};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

pub struct AnnpTokenizer {
    inner: SentencePieceProcessor,
}

impl AnnpTokenizer {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Self {
        let p = path.as_ref();

        match SentencePieceProcessor::open(p.to_str().unwrap()) {
            Ok(spp) => {
                println!("Successfully loaded SentencePiece Tokenizer from: {:?}", p);
                Self { inner: spp }
            }
            Err(e) => panic!(
                "CRITICAL ERROR: Failed to load tokenizer file {:?}: {}",
                p, e
            ),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Ok(pieces) = self.inner.encode(text) {
            pieces.into_iter().map(|p| p.id).collect()
        } else {
            Vec::new()
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode_piece_ids(ids).unwrap_or_default()
    }

    pub fn encode_ids_to_tensor(
        &self,
        ids: &[u32],
        d_model: usize,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let seq_len = ids.len().max(1);
        let mut flat = Vec::with_capacity(seq_len * d_model);

        for (pos, &token_id) in ids.iter().enumerate() {
            let mut tok_vec = Vec::with_capacity(d_model);
            let mut seed = (token_id as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(1);

            for _ in 0..d_model {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let rand_f32 = ((seed & 0xFFFFFFFF) as f32 / 4294967295.0f32) * 2.0f32 - 1.0f32;
                tok_vec.push(rand_f32);
            }

            // RMS Normalization to anchor unit embedding scale (~1.0)
            let rms = (tok_vec.iter().map(|v| v * v).sum::<f32>() / d_model as f32)
                .sqrt()
                .max(1e-6);
            for d in 0..d_model {
                // Positional Encoding *after* RMSNorm
                let pos_enc = (pos as f32 * 0.05f32 + d as f32 * 0.01f32).sin() * 0.1f32;
                flat.push(tok_vec[d] / rms + pos_enc);
            }
        }

        Tensor::from_vec(flat, (seq_len, d_model), device)
    }

    /// Convert tokenized sequence into dense ANNP Input Activation Tensor [seq_len, d_model]
    pub fn encode_to_tensor(
        &self,
        text: &str,
        d_model: usize,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let ids = self.encode(text);
        self.encode_ids_to_tensor(&ids, d_model, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode() {
        let tokenizer = AnnpTokenizer::load_from_file("../tokenizer.model");
        let text = "ANNP Asynchronous Neural Network Protocol Simulation";
        let ids = tokenizer.encode(text);
        assert!(!ids.is_empty());
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.is_empty());
    }
}
