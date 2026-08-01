/// Mathematical metrics for Halting conditions and Attention Entropy evaluation.
///
/// OnlineStats implements an Exponential-decay Welford algorithm.
/// The decay factor `gamma` serves two roles:
///   1. Exponentially decays old M2 (so old variance contributions fade)
///   2. Determines the effective window via count decay: count_t = gamma*count_{t-1} + 1
///      which converges to 1/(1-gamma) — for gamma=0.99, effective_N ≈ 100.
///
/// The key property: alpha = 1/count_t starts at 1.0 (first obs sets mean exactly)
/// and decays toward 1-gamma = 0.01. This is essential for Thompson Sampling to
/// quickly differentiate subnodes after neurogenesis.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct OnlineStats {
    pub count: f32,
    pub mean: f32,
    /// Exponential-decay M2 accumulator. variance() = m2/count.
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

    /// Update using Exponential-decay Welford:
    ///   count_t = gamma * count_{t-1} + 1          (converges to 1/(1-gamma))
    ///   mean_t  = mean_{t-1} + (x - mean_{t-1}) / count_t   (effective lr = 1/count_t)
    ///   M2_t    = gamma * M2_{t-1} + (x - mean_{t-1}) * (x - mean_t)
    ///   var_t   = M2_t / count_t
    ///
    /// The 1/count_t lr starts at 1.0 (first observation) and decays toward
    /// 1-gamma = 0.01. This enables Thompson Sampling to rapidly evaluate
    /// newly spawned subnodes without 100-step warm-up.
    pub fn observe(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.count = self.count * self.gamma + 1.0;
        let delta = value - self.mean;
        self.mean += delta / self.count;
        self.m2 = self.m2 * self.gamma + delta * (value - self.mean);
    }

    /// Variance estimate: M2 / count.
    pub fn variance(&self) -> f32 {
        if self.count > 0.0 {
            (self.m2 / self.count).max(0.0)
        } else {
            0.0
        }
    }

    /// Standard error of the mean: SE = sqrt(variance / count) = sqrt(M2 / count²).
    /// For effective_N ≈ 100, this gives SE ≈ std/10, matching the classical formula.
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
pub fn student_t_sample_approximation(df: f32, u1: f32, u2: f32) -> f32 {
    let u1 = u1.max(f32::MIN_POSITIVE);
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
    let epsilon = RMS_EPSILON;
    let mut entropy = 0.0f32;
    for &p in probs {
        if p > epsilon {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Shared epsilon constant for numerical stability in RMSNorm and entropy calculations.
/// Using 1e-8 throughout to avoid inconsistency between metrics.rs (1e-5) and
/// micro_block.rs (1e-8). The smaller value is safer for f32 computations.
pub const RMS_EPSILON: f32 = 1e-8;

/// Sphere normalization helper: p_out = (p_in + sublayer_out) / (||...||_2 + eps) * S_base
pub fn sphere_normalize(vec: &mut [f32], radius: f32) {
    let norm_sq: f32 = vec.iter().map(|&x| x * x).sum();
    let norm = (norm_sq + RMS_EPSILON).sqrt();
    let scale = radius / norm;
    for x in vec.iter_mut() {
        *x *= scale;
    }
}

/// Micro-RMSNorm helper: output = input / RMS(input)
/// Uses shared RMS_EPSILON for consistency with micro_block.rs inline implementations.
pub fn rms_normalize(sublayer: &[f32]) -> Vec<f32> {
    let n = sublayer.len() as f32;
    let mean_sq: f32 = sublayer.iter().map(|&x| x * x).sum::<f32>() / n;
    let rms = (mean_sq + RMS_EPSILON).sqrt();
    sublayer.iter().map(|&x| x / rms).collect()
}
