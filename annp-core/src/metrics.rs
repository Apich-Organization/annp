/// Mathematical metrics for Halting conditions and Attention Entropy evaluation.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct OnlineStats {
    pub count: f32,
    pub mean: f32,
    pub m2: f32,
    pub gamma: f32,
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
            gamma: 0.99,
        }
    }
}

impl OnlineStats {
    pub fn new(gamma: f32) -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
            gamma,
        }
    }

    pub fn observe(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.count = self.count * self.gamma + 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.count;
        self.m2 = self.m2 * self.gamma + delta * (value - self.mean);
    }

    pub fn variance(&self) -> f32 {
        if self.count > 1.0 {
            (self.m2 / (self.count - 1.0)).max(0.0)
        } else {
            0.0
        }
    }

    pub fn standard_error(&self) -> f32 {
        if self.count > 1.0 {
            (self.variance() / self.count).sqrt()
        } else {
            f32::INFINITY
        }
    }

    pub fn decay(&mut self) {
        self.count *= self.gamma;
        self.m2 *= self.gamma;
    }
}

/// Approximate sampling from a Student-t distribution with `df` degrees of freedom
/// using 1st order Cornish-Fisher expansion.
pub fn student_t_sample_approximation(df: f32) -> f32 {
    let u1: f32 = rand::random::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rand::random::<f32>();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

    if df < 1.0 {
        return z0; // Fallback
    }

    // Cornish-Fisher expansion for heavier tails of Student-t
    z0 * (1.0 + (z0 * z0 + 1.0) / (4.0 * df))
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
pub fn compute_memory_density(probs: &[f32]) -> f32 {
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
