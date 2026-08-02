use annp_core::RMS_EPSILON;
use candle_core::{Device, Result as CandleResult, Tensor};
use sentencepiece::SentencePieceProcessor;
use std::path::Path;

/// Fundamental Fourier base frequency for positional encodings
pub const POS_BASE_FREQ: f32 = 0.05;
/// Inter-dimension spatial dispersion frequency
pub const DIM_BASE_FREQ: f32 = 0.01;
/// Positional encoding amplitude modulation scale
pub const POS_ENC_SCALE: f32 = 0.1;

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

    /// Return total vocabulary size from SentencePiece model
    pub fn vocab_size(&self) -> usize {
        self.inner.len()
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

    /// Generate deterministic embedding vector for a token at a given sequence position
    pub fn token_embedding(token_id: u32, pos: usize, d_model: usize) -> Vec<f32> {
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

        // RMS Normalization using shared RMS_EPSILON to anchor unit embedding scale (~1.0)
        let mean_sq: f32 = tok_vec.iter().map(|v| v * v).sum::<f32>() / d_model as f32;
        let rms = (mean_sq + RMS_EPSILON).sqrt();
        let mut out = Vec::with_capacity(d_model);
        for (d, &tok_val) in tok_vec.iter().enumerate().take(d_model) {
            let pos_enc =
                (pos as f32 * POS_BASE_FREQ + d as f32 * DIM_BASE_FREQ).sin() * POS_ENC_SCALE;
            out.push(tok_val / rms + pos_enc);
        }
        out
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
            flat.extend(Self::token_embedding(token_id, pos, d_model));
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
    fn test_tokenizer_token_embedding() {
        let emb1 = AnnpTokenizer::token_embedding(42, 0, 64);
        assert_eq!(emb1.len(), 64);
        let emb2 = AnnpTokenizer::token_embedding(42, 0, 64);
        assert_eq!(emb1, emb2);

        // Verify different positions produce distinct encodings
        let emb_pos1 = AnnpTokenizer::token_embedding(42, 1, 64);
        assert_ne!(emb1, emb_pos1);
    }
}
