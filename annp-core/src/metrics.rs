/// Mathematical metrics for Halting conditions and Attention Entropy evaluation.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct OnlineStats {
    pub count: u64,
    pub mean: f32,
    pub m2: f32,
}

impl OnlineStats {
    pub fn observe(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        self.m2 += delta * (value - self.mean);
    }

    pub fn variance(&self) -> f32 {
        if self.count > 1 {
            (self.m2 / (self.count - 1) as f32).max(0.0)
        } else {
            0.0
        }
    }

    pub fn standard_error(&self) -> f32 {
        if self.count > 1 {
            (self.variance() / self.count as f32).sqrt()
        } else {
            f32::INFINITY
        }
    }

    pub fn optimistic_value(&self) -> f32 {
        self.mean + self.standard_error()
    }
    pub fn pessimistic_value(&self) -> f32 {
        self.mean - self.standard_error()
    }
}

/// Compute L2 norm difference between input and output particle payloads: ||p_out - p_in||_2
pub fn compute_delta_p(p_in: &[f32], p_out: &[f32]) -> f32 {
    assert_eq!(
        p_in.len(),
        p_out.len(),
        "Particle dimension mismatch in delta_p computation"
    );
    let sum_sq: f32 = p_in
        .iter()
        .zip(p_out.iter())
        .map(|(&x, &y)| {
            let diff = y - x;
            diff * diff
        })
        .sum();
    sum_sq.sqrt()
}

/// Compute Shannon Entropy H(probs) over attention probability distribution vector.
/// H = - \sum_i p_i \log_2(p_i + \epsilon)
pub fn compute_attention_entropy(probs: &[f32]) -> f32 {
    let epsilon = 1e-12f32;
    let mut entropy = 0.0f32;
    for &p in probs {
        if p > epsilon {
            entropy -= p * (p + epsilon).log2();
        }
    }
    entropy
}

/// Sphere normalization helper: p_out = (p_in + sublayer_out) / (||...||_2 + eps) * S_base
pub fn sphere_normalize(vec: &mut [f32], radius: f32) {
    let norm_sq: f32 = vec.iter().map(|&x| x * x).sum();
    let norm = (norm_sq + 1e-8).sqrt();
    let scale = radius / norm;
    for x in vec.iter_mut() {
        *x *= scale;
    }
}

/// Micro-RMSNorm helper with alpha scaling: p_out = p_in + alpha * RMSNorm(sublayer)
pub fn rms_normalize(sublayer: &[f32]) -> Vec<f32> {
    let n = sublayer.len() as f32;
    let mean_sq: f32 = sublayer.iter().map(|&x| x * x).sum::<f32>() / n;
    let rms = (mean_sq + 1e-8).sqrt();
    sublayer.iter().map(|&x| x / rms).collect()
}
