#[cfg(cuda_available)]
use std::ffi::c_void;

use crate::stream::CudaStreamManager;

pub const MAX_D_HEAD: usize = 256;
pub const MAX_FFN_DIM: usize = 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleCudaHeader {
    pub origin_token_id: u32,
    pub shard_id: u16,
    pub pad0: [u8; 2],
    pub energy: f32,
    pub hop_count: u16,
    pub halted: u8,
    pub pad1: [u8; 1],
}

const _: () = assert!(std::mem::size_of::<ParticleCudaHeader>() == 16);

impl ParticleCudaHeader {
    pub fn is_halted(&self) -> bool {
        self.halted != 0
    }
    pub fn set_halted(&mut self, h: bool) {
        self.halted = if h { 1 } else { 0 };
    }
}

unsafe extern "C" {
    fn launch_fused_micro_block_backward(
        p_in: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        w_gate: *mut f32,
        w_up: *mut f32,
        w_down: *mut f32,
        local_err: *const f32,
        d_head: i32,
        ffn_dim: i32,
        kv_len: i32,
        alpha: f32,
        lr: f32,
        weight_decay: f32,
        stream: *mut std::ffi::c_void,
    );
}

#[cfg(cuda_available)]
unsafe extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(devPtr: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;

    pub fn launch_fused_micro_block(
        p_in: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        w_gate: *const f32,
        w_up: *const f32,
        w_down: *const f32,
        p_out: *mut f32,
        batch_size: i32,
        d_head: i32,
        ffn_dim: i32,
        kv_len: i32,
        norm_strategy: i32,
        alpha: f32,
        sphere_radius: f32,
        stream: *mut c_void,
    );

    pub fn launch_particle_router(
        p_in: *const f32,
        p_out: *const f32,
        routing_table: *const f32,
        gumbel_noise: *const f32,
        chosen_neighbor: *mut i32,
        halting_flags: *mut bool,
        batch_size: i32,
        d_head: i32,
        num_neighbors: i32,
        temperature: f32,
        epsilon_p: f32,
        epsilon_h: f32,
        min_hop: i32,
        headers: *const ParticleCudaHeader,
        stream: *mut c_void,
    );
}

/// Safe Rust interface wrapper for Fused Micro-Block Computation.
pub struct CudaMicroBlockRunner;

impl CudaMicroBlockRunner {
    pub fn execute_fused(
        p_in: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        ffn_dim: usize,
        kv_len: usize,
        alpha: f32,
        d_weights: Option<&CudaDeviceWeights>,
    ) {
        Self::execute_fused_with_stream_device(
            p_in, k_cache, v_cache, w_gate, w_up, w_down, p_out, batch_size, d_head, ffn_dim,
            kv_len, alpha, None, true, d_weights,
        );
    }

    pub fn execute_fused_with_stream(
        p_in: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        ffn_dim: usize,
        kv_len: usize,
        alpha: f32,
        stream: Option<&CudaStreamManager>,
        d_weights: Option<&CudaDeviceWeights>,
    ) {
        Self::execute_fused_with_stream_device(
            p_in, k_cache, v_cache, w_gate, w_up, w_down, p_out, batch_size, d_head, ffn_dim,
            kv_len, alpha, stream, true, d_weights,
        );
    }

    #[allow(unused_variables)]
    pub fn execute_fused_with_stream_device(
        p_in: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        ffn_dim: usize,
        kv_len: usize,
        alpha: f32,
        stream: Option<&CudaStreamManager>,
        use_cuda: bool,
        d_weights: Option<&CudaDeviceWeights>,
    ) {
        if batch_size == 0 || d_head == 0 || ffn_dim == 0 {
            return;
        }

        assert!(batch_size <= i32::MAX as usize);
        assert!(d_head <= i32::MAX as usize);
        assert!(ffn_dim <= i32::MAX as usize);
        assert!(kv_len <= i32::MAX as usize);

        // Bounds Safety Validation
        assert!(
            p_in.len() >= batch_size * d_head,
            "p_in slice too small: expected {}, got {}",
            batch_size * d_head,
            p_in.len()
        );
        assert!(
            p_out.len() >= batch_size * d_head,
            "p_out slice too small: expected {}, got {}",
            batch_size * d_head,
            p_out.len()
        );
        if kv_len > 0 {
            assert!(
                k_cache.len() >= kv_len * d_head,
                "k_cache slice too small: expected {}, got {}",
                kv_len * d_head,
                k_cache.len()
            );
            assert!(
                v_cache.len() >= kv_len * d_head,
                "v_cache slice too small: expected {}, got {}",
                kv_len * d_head,
                v_cache.len()
            );
        }
        assert!(
            w_gate.len() >= d_head * ffn_dim,
            "w_gate slice too small: expected {}, got {}",
            d_head * ffn_dim,
            w_gate.len()
        );
        assert!(
            w_up.len() >= d_head * ffn_dim,
            "w_up slice too small: expected {}, got {}",
            d_head * ffn_dim,
            w_up.len()
        );
        assert!(
            w_down.len() >= ffn_dim * d_head,
            "w_down slice too small: expected {}, got {}",
            ffn_dim * d_head,
            w_down.len()
        );

        if use_cuda {
            #[cfg(cuda_available)]
            unsafe {
                let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.stream_ptr());
                let w_gate_ptr = d_weights.map_or(w_gate.as_ptr(), |d| d.d_w_gate as *const f32);
                let w_up_ptr = d_weights.map_or(w_up.as_ptr(), |d| d.d_w_up as *const f32);
                let w_down_ptr = d_weights.map_or(w_down.as_ptr(), |d| d.d_w_down as *const f32);

                launch_fused_micro_block(
                    p_in.as_ptr(),
                    k_cache.as_ptr(),
                    v_cache.as_ptr(),
                    w_gate_ptr,
                    w_up_ptr,
                    w_down_ptr,
                    p_out.as_mut_ptr(),
                    batch_size as i32,
                    d_head as i32,
                    ffn_dim as i32,
                    kv_len as i32,
                    0, // norm_strategy (0 = RMSNorm)
                    alpha,
                    1.0, // sphere_radius
                    stream_ptr,
                );
                return;
            }
        }

        #[cfg(target_arch = "x86_64")]
        let is_x86_avx2 = d_head <= MAX_D_HEAD
            && d_head.is_multiple_of(8)
            && ffn_dim <= MAX_FFN_DIM
            && ffn_dim.is_multiple_of(8)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma");

        #[cfg(not(target_arch = "x86_64"))]
        let is_x86_avx2 = false;

        if is_x86_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                Self::execute_fused_avx2(
                    p_in, k_cache, v_cache, w_gate, w_up, w_down, p_out, batch_size, d_head,
                    ffn_dim, kv_len, alpha,
                );
            }
        } else {
            Self::execute_fused_fallback(
                p_in, k_cache, v_cache, w_gate, w_up, w_down, p_out, batch_size, d_head, ffn_dim,
                kv_len, alpha,
            );
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[cfg(target_arch = "x86_64")]
    unsafe fn execute_fused_avx2(
        p_in: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        ffn_dim: usize,
        kv_len: usize,
        alpha: f32,
    ) {
        unsafe {
            use std::arch::x86_64::*;

            assert!(
                d_head <= MAX_D_HEAD && d_head.is_multiple_of(8),
                "d_head must be <= MAX_D_HEAD and divisible by 8"
            );
            assert!(
                ffn_dim <= MAX_FFN_DIM && ffn_dim.is_multiple_of(8),
                "ffn_dim must be <= MAX_FFN_DIM and divisible by 8"
            );
            assert!(p_in.len() >= batch_size * d_head);
            assert!(p_out.len() >= batch_size * d_head);

            #[inline]
            #[target_feature(enable = "avx2,fma")]
            unsafe fn hsum_avx2(v: __m256) -> f32 {
                let vlow = _mm256_castps256_ps128(v);
                let vhigh = _mm256_extractf128_ps(v, 1);
                let v128 = _mm_add_ps(vlow, vhigh);
                let shuf = _mm_movehdup_ps(v128);
                let sums = _mm_add_ps(v128, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let sums2 = _mm_add_ss(sums, shuf2);
                _mm_cvtss_f32(sums2)
            }

            let scale = 1.0 / (d_head as f32).sqrt();

            for b in 0..batch_size {
                let curr_p = &p_in[b * d_head..(b + 1) * d_head];
                let out_slice = &mut p_out[b * d_head..(b + 1) * d_head];

                let mut attn_out = [0.0f32; MAX_D_HEAD];
                let mut s_mid = [0.0f32; MAX_D_HEAD];
                let mut ffn_inter = [0.0f32; MAX_FFN_DIM];

                // 1. Pre-RMSNorm for Attention
                let mut curr_p_normed = [0.0f32; MAX_D_HEAD];
                let mut sq_sum_attn_v = _mm256_setzero_ps();
                for d in (0..d_head).step_by(8) {
                    let p_v = _mm256_loadu_ps(curr_p.as_ptr().add(d));
                    sq_sum_attn_v = _mm256_fmadd_ps(p_v, p_v, sq_sum_attn_v);
                }
                let sq_sum_attn = hsum_avx2(sq_sum_attn_v);
                let inv_rms_attn = 1.0 / (sq_sum_attn / (d_head as f32) + 1e-8).sqrt();
                let inv_rms_attn_v = _mm256_set1_ps(inv_rms_attn);
                for d in (0..d_head).step_by(8) {
                    let p_v = _mm256_loadu_ps(curr_p.as_ptr().add(d));
                    let normed = _mm256_mul_ps(p_v, inv_rms_attn_v);
                    _mm256_storeu_ps(curr_p_normed.as_mut_ptr().add(d), normed);
                }

                // 1.5 Full KV-Attention Dot Product
                if kv_len > 0 {
                    let mut scores = vec![0.0f32; kv_len];
                    let mut max_score = -1e9f32;

                    for k in 0..kv_len {
                        let k_slice = &k_cache[k * d_head..(k + 1) * d_head];
                        let mut acc_v = _mm256_setzero_ps();

                        for d in (0..d_head).step_by(8) {
                            let p_vec = _mm256_loadu_ps(curr_p_normed.as_ptr().add(d));
                            let k_vec = _mm256_loadu_ps(k_slice.as_ptr().add(d));
                            acc_v = _mm256_fmadd_ps(p_vec, k_vec, acc_v);
                        }

                        let dot = hsum_avx2(acc_v) * scale;
                        scores[k] = dot;
                        if dot > max_score {
                            max_score = dot;
                        }
                    }

                    let mut sum_exp = 0.0f32;
                    for k in 0..kv_len {
                        let e = (scores[k] - max_score).exp();
                        scores[k] = e;
                        sum_exp += e;
                    }

                    let inv_sum = 1.0 / (sum_exp + 1e-8);
                    for k in 0..kv_len {
                        let w = scores[k] * inv_sum;
                        let v_slice = &v_cache[k * d_head..(k + 1) * d_head];
                        let w_vec = _mm256_set1_ps(w);

                        for d in (0..d_head).step_by(8) {
                            let v_vec = _mm256_loadu_ps(v_slice.as_ptr().add(d));
                            let curr_attn = _mm256_loadu_ps(attn_out.as_ptr().add(d));
                            let res = _mm256_fmadd_ps(w_vec, v_vec, curr_attn);
                            _mm256_storeu_ps(attn_out.as_mut_ptr().add(d), res);
                        }
                    }
                }

                // 2. Additive Residual 1
                let alpha_vec = _mm256_set1_ps(alpha);
                for d in (0..d_head).step_by(8) {
                    let p_v = _mm256_loadu_ps(curr_p.as_ptr().add(d));
                    let a_v = _mm256_loadu_ps(attn_out.as_ptr().add(d));
                    let res = _mm256_fmadd_ps(a_v, alpha_vec, p_v);
                    _mm256_storeu_ps(s_mid.as_mut_ptr().add(d), res);
                }

                // 3. Pre-RMSNorm for FFN
                let mut s_mid_normed = [0.0f32; MAX_D_HEAD];
                let mut sq_sum_ffn_v = _mm256_setzero_ps();
                for d in (0..d_head).step_by(8) {
                    let s_v = _mm256_loadu_ps(s_mid.as_ptr().add(d));
                    sq_sum_ffn_v = _mm256_fmadd_ps(s_v, s_v, sq_sum_ffn_v);
                }
                let sq_sum_ffn = hsum_avx2(sq_sum_ffn_v);
                let inv_rms_ffn = 1.0 / (sq_sum_ffn / (d_head as f32) + 1e-8).sqrt();
                let inv_rms_ffn_v = _mm256_set1_ps(inv_rms_ffn);
                for d in (0..d_head).step_by(8) {
                    let s_v = _mm256_loadu_ps(s_mid.as_ptr().add(d));
                    let normed = _mm256_mul_ps(s_v, inv_rms_ffn_v);
                    _mm256_storeu_ps(s_mid_normed.as_mut_ptr().add(d), normed);
                }

                // 3.5 SwiGLU FFN
                let mut gate_arr = [0.0f32; MAX_FFN_DIM];
                let mut up_arr = [0.0f32; MAX_FFN_DIM];

                for d in 0..d_head {
                    let m_v = _mm256_set1_ps(s_mid_normed[d]);
                    let d_offset = d * ffn_dim;

                    for j in (0..ffn_dim).step_by(8) {
                        let wg_v = _mm256_loadu_ps(w_gate.as_ptr().add(d_offset + j));
                        let curr_g = _mm256_loadu_ps(gate_arr.as_ptr().add(j));
                        _mm256_storeu_ps(
                            gate_arr.as_mut_ptr().add(j),
                            _mm256_fmadd_ps(m_v, wg_v, curr_g),
                        );

                        let wu_v = _mm256_loadu_ps(w_up.as_ptr().add(d_offset + j));
                        let curr_u = _mm256_loadu_ps(up_arr.as_ptr().add(j));
                        _mm256_storeu_ps(
                            up_arr.as_mut_ptr().add(j),
                            _mm256_fmadd_ps(m_v, wu_v, curr_u),
                        );
                    }
                }

                for j in 0..ffn_dim {
                    let gate = gate_arr[j];
                    let up = up_arr[j];
                    let swish = gate / (1.0 + (-gate).exp());
                    ffn_inter[j] = swish * up;
                }

                // 4. Down Projection
                let mut down_arr = [0.0f32; MAX_D_HEAD];

                for j in 0..ffn_dim {
                    let inter_v = _mm256_set1_ps(ffn_inter[j]);
                    let j_offset = j * d_head;

                    for d in (0..d_head).step_by(8) {
                        let wd_v = _mm256_loadu_ps(w_down.as_ptr().add(j_offset + d));
                        let curr_d = _mm256_loadu_ps(down_arr.as_ptr().add(d));
                        _mm256_storeu_ps(
                            down_arr.as_mut_ptr().add(d),
                            _mm256_fmadd_ps(inter_v, wd_v, curr_d),
                        );
                    }
                }

                // 5. Additive Residual 2 & Output Clipping
                let min_v = _mm256_set1_ps(-100.0);
                let max_v = _mm256_set1_ps(100.0);
                let alpha_v = _mm256_set1_ps(alpha);

                for d in (0..d_head).step_by(8) {
                    let s_mid_v = _mm256_loadu_ps(s_mid.as_ptr().add(d));
                    let down_v = _mm256_loadu_ps(down_arr.as_ptr().add(d));
                    let res = _mm256_fmadd_ps(down_v, alpha_v, s_mid_v);
                    let clamped = _mm256_min_ps(_mm256_max_ps(res, min_v), max_v);
                    _mm256_storeu_ps(out_slice.as_mut_ptr().add(d), clamped);
                }
            }

            // Zero upper 128-bit YMM register states to prevent AVX-SSE transition penalty
            _mm256_zeroupper();
        }
    }

    fn execute_fused_fallback(
        p_in: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        ffn_dim: usize,
        kv_len: usize,
        alpha: f32,
    ) {
        let scale = 1.0 / (d_head as f32).sqrt();

        for b in 0..batch_size {
            let curr_p = &p_in[b * d_head..(b + 1) * d_head];
            let out_slice = &mut p_out[b * d_head..(b + 1) * d_head];

            // 1. Pre-RMSNorm for Attention
            let mut curr_p_normed = vec![0.0f32; d_head];
            let sq_sum_attn: f32 = curr_p.iter().map(|&x| x * x).sum();
            let inv_rms_attn = 1.0 / (sq_sum_attn / (d_head as f32) + 1e-8).sqrt();
            for d in 0..d_head {
                curr_p_normed[d] = curr_p[d] * inv_rms_attn;
            }

            // 1.5 Attention
            let mut attn_out = vec![0.0f32; d_head];
            if kv_len > 0 && !k_cache.is_empty() && !v_cache.is_empty() {
                let actual_kv = kv_len
                    .min(k_cache.len() / d_head)
                    .min(v_cache.len() / d_head);
                if actual_kv > 0 {
                    let mut scores = vec![0.0f32; actual_kv];
                    let mut max_score = f32::NEG_INFINITY;

                    for k in 0..actual_kv {
                        let k_slice = &k_cache[k * d_head..(k + 1) * d_head];
                        let mut dot = 0.0f32;
                        for d in 0..d_head {
                            dot += curr_p_normed[d] * k_slice[d];
                        }
                        let score = dot * scale;
                        scores[k] = score;
                        if score > max_score {
                            max_score = score;
                        }
                    }

                    let mut sum_exp = 0.0f32;
                    for k in 0..actual_kv {
                        let e = (scores[k] - max_score).exp();
                        scores[k] = e;
                        sum_exp += e;
                    }

                    let inv_sum = 1.0 / (sum_exp + 1e-8);
                    for k in 0..actual_kv {
                        let w = scores[k] * inv_sum;
                        let v_slice = &v_cache[k * d_head..(k + 1) * d_head];
                        for d in 0..d_head {
                            attn_out[d] += w * v_slice[d];
                        }
                    }
                }
            }

            // 2. Additive Residual 1
            let mut s_mid = vec![0.0f32; d_head];
            for d in 0..d_head {
                s_mid[d] = curr_p[d] + alpha * attn_out[d];
            }

            // 3. Pre-RMSNorm for FFN
            let mut s_mid_normed = vec![0.0f32; d_head];
            let sq_sum_ffn: f32 = s_mid.iter().map(|&x| x * x).sum();
            let inv_rms_ffn = 1.0 / (sq_sum_ffn / (d_head as f32) + 1e-8).sqrt();
            for d in 0..d_head {
                s_mid_normed[d] = s_mid[d] * inv_rms_ffn;
            }

            // 3.5 SwiGLU FFN
            let mut ffn_inter = vec![0.0f32; ffn_dim];
            for j in 0..ffn_dim {
                let mut gate = 0.0f32;
                let mut up = 0.0f32;
                for d in 0..d_head {
                    let m_val = s_mid_normed[d];
                    gate += m_val * w_gate[d * ffn_dim + j];
                    up += m_val * w_up[d * ffn_dim + j];
                }
                let swish = gate / (1.0 + (-gate).exp());
                ffn_inter[j] = swish * up;
            }

            // 4. Down Projection
            let mut down_arr = vec![0.0f32; d_head];
            for j in 0..ffn_dim {
                let inter_val = ffn_inter[j];
                for d in 0..d_head {
                    down_arr[d] += inter_val * w_down[j * d_head + d];
                }
            }

            // 5. Additive Residual 2 & Output Clipping
            for d in 0..d_head {
                let res = s_mid[d] + alpha * down_arr[d];
                out_slice[d] = res.clamp(-100.0, 100.0);
            }
        }
    }

    #[allow(unused_variables)]
    pub fn execute_backward(
        p_in: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        w_gate: &mut [f32],
        w_up: &mut [f32],
        w_down: &mut [f32],
        local_err: &[f32],
        d_head: usize,
        ffn_dim: usize,
        kv_len: usize,
        alpha: f32,
        lr: f32,
        weight_decay: f32,
        stream: Option<&CudaStreamManager>,
        use_cuda: bool,
        d_weights: Option<&CudaDeviceWeights>,
    ) {
        if d_head == 0 || ffn_dim == 0 {
            return;
        }

        if use_cuda {
            #[cfg(cuda_available)]
            unsafe {
                let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.stream_ptr());
                let w_gate_ptr = d_weights.map_or(w_gate.as_mut_ptr(), |d| d.d_w_gate as *mut f32);
                let w_up_ptr = d_weights.map_or(w_up.as_mut_ptr(), |d| d.d_w_up as *mut f32);
                let w_down_ptr = d_weights.map_or(w_down.as_mut_ptr(), |d| d.d_w_down as *mut f32);

                launch_fused_micro_block_backward(
                    p_in.as_ptr(),
                    k_cache.as_ptr(),
                    v_cache.as_ptr(),
                    w_gate_ptr,
                    w_up_ptr,
                    w_down_ptr,
                    local_err.as_ptr(),
                    d_head as i32,
                    ffn_dim as i32,
                    kv_len as i32,
                    alpha,
                    lr,
                    weight_decay,
                    stream_ptr,
                );
            }
        }
    }
}

/// Safe Rust interface wrapper for Particle Q-Router Computation.
pub struct CudaParticleRouter;

impl CudaParticleRouter {
    pub fn execute_routing(
        p_in: &[f32],
        p_out: &[f32],
        routing_table: &[f32],
        gumbel_noise: &[f32],
        chosen_neighbor: &mut [usize],
        halting_flags: &mut [bool],
        batch_size: usize,
        d_head: usize,
        num_neighbors: usize,
        temperature: f32,
        epsilon_p: f32,
        epsilon_h: f32,
        min_hop: u16,
        headers: &[ParticleCudaHeader],
    ) {
        Self::execute_routing_with_stream_device(
            p_in,
            p_out,
            routing_table,
            gumbel_noise,
            chosen_neighbor,
            halting_flags,
            batch_size,
            d_head,
            num_neighbors,
            temperature,
            epsilon_p,
            epsilon_h,
            min_hop,
            headers,
            None,
            true,
        );
    }

    pub fn execute_routing_with_stream(
        p_in: &[f32],
        p_out: &[f32],
        routing_table: &[f32],
        gumbel_noise: &[f32],
        chosen_neighbor: &mut [usize],
        halting_flags: &mut [bool],
        batch_size: usize,
        d_head: usize,
        num_neighbors: usize,
        temperature: f32,
        epsilon_p: f32,
        epsilon_h: f32,
        min_hop: u16,
        headers: &[ParticleCudaHeader],
        stream: Option<&CudaStreamManager>,
    ) {
        Self::execute_routing_with_stream_device(
            p_in,
            p_out,
            routing_table,
            gumbel_noise,
            chosen_neighbor,
            halting_flags,
            batch_size,
            d_head,
            num_neighbors,
            temperature,
            epsilon_p,
            epsilon_h,
            min_hop,
            headers,
            stream,
            true,
        );
    }

    #[allow(unused_variables)]
    pub fn execute_routing_with_stream_device(
        p_in: &[f32],
        p_out: &[f32],
        routing_table: &[f32],
        gumbel_noise: &[f32],
        chosen_neighbor: &mut [usize],
        halting_flags: &mut [bool],
        batch_size: usize,
        d_head: usize,
        num_neighbors: usize,
        temperature: f32,
        epsilon_p: f32,
        epsilon_h: f32,
        min_hop: u16,
        headers: &[ParticleCudaHeader],
        stream: Option<&CudaStreamManager>,
        use_cuda: bool,
    ) {
        assert!(batch_size <= i32::MAX as usize);
        assert!(d_head <= i32::MAX as usize);
        assert!(num_neighbors <= i32::MAX as usize);

        assert!(
            chosen_neighbor.len() >= batch_size,
            "chosen_neighbor slice too small: expected {}, got {}",
            batch_size,
            chosen_neighbor.len()
        );
        assert!(
            halting_flags.len() >= batch_size,
            "halting_flags slice too small: expected {}, got {}",
            batch_size,
            halting_flags.len()
        );

        if batch_size == 0 {
            return;
        }

        if d_head == 0 || num_neighbors == 0 {
            for dst in chosen_neighbor.iter_mut().take(batch_size) {
                *dst = 0;
            }
            for dst in halting_flags.iter_mut().take(batch_size) {
                *dst = false;
            }
            return;
        }

        assert!(
            p_in.len() >= batch_size * d_head,
            "p_in slice too small: expected {}, got {}",
            batch_size * d_head,
            p_in.len()
        );
        assert!(
            p_out.len() >= batch_size * d_head,
            "p_out slice too small: expected {}, got {}",
            batch_size * d_head,
            p_out.len()
        );
        assert!(
            routing_table.len() >= d_head * num_neighbors,
            "routing_table slice too small: expected {}, got {}",
            d_head * num_neighbors,
            routing_table.len()
        );
        assert!(
            gumbel_noise.len() >= batch_size * num_neighbors,
            "gumbel_noise slice too small: expected {}, got {}",
            batch_size * num_neighbors,
            gumbel_noise.len()
        );
        assert!(
            headers.len() >= batch_size,
            "headers slice too small: expected {}, got {}",
            batch_size,
            headers.len()
        );

        if use_cuda {
            #[cfg(cuda_available)]
            {
                let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.stream_ptr());
                let mut chosen_i32 = vec![0i32; batch_size];
                let mut halting_b = vec![false; batch_size];

                unsafe {
                    launch_particle_router(
                        p_in.as_ptr(),
                        p_out.as_ptr(),
                        routing_table.as_ptr(),
                        gumbel_noise.as_ptr(),
                        chosen_i32.as_mut_ptr(),
                        halting_b.as_mut_ptr(),
                        batch_size as i32,
                        d_head as i32,
                        num_neighbors as i32,
                        temperature,
                        epsilon_p,
                        epsilon_h,
                        min_hop as i32,
                        headers.as_ptr(),
                        stream_ptr,
                    );
                }

                for (dst, &src) in chosen_neighbor.iter_mut().zip(chosen_i32.iter()) {
                    *dst = src.max(0) as usize;
                }
                halting_flags[..batch_size].copy_from_slice(&halting_b[..batch_size]);
                return;
            }
        }

        Self::execute_routing_fallback(
            p_in,
            p_out,
            routing_table,
            gumbel_noise,
            chosen_neighbor,
            halting_flags,
            batch_size,
            d_head,
            num_neighbors,
            temperature,
            epsilon_p,
            epsilon_h,
            min_hop,
            headers,
        );
    }

    fn execute_routing_fallback(
        p_in: &[f32],
        p_out: &[f32],
        routing_table: &[f32],
        gumbel_noise: &[f32],
        chosen_neighbor: &mut [usize],
        halting_flags: &mut [bool],
        batch_size: usize,
        d_head: usize,
        num_neighbors: usize,
        temperature: f32,
        epsilon_p: f32,
        epsilon_h: f32,
        min_hop: u16,
        headers: &[ParticleCudaHeader],
    ) {
        let inv_temp = 1.0f32 / temperature.max(1e-4f32);
        let noise_base = if num_neighbors > 0 {
            0.05f32 / (num_neighbors as f32)
        } else {
            0.0f32
        };

        for b in 0..batch_size {
            let pin_slice = &p_in[b * d_head..(b + 1) * d_head];
            let pout_slice = &p_out[b * d_head..(b + 1) * d_head];
            let gnoise_slice = &gumbel_noise[b * num_neighbors..(b + 1) * num_neighbors];

            // 1. Delta p
            let sum_sq: f32 = pin_slice
                .iter()
                .zip(pout_slice.iter())
                .map(|(&x, &y)| {
                    let d = y - x;
                    d * d
                })
                .sum();
            let delta_p = sum_sq.sqrt();

            // 2. Logits & Softmax (Matches layout [d_head, num_neighbors])
            let mut logits = vec![0.0f32; num_neighbors];
            let mut max_logit = f32::NEG_INFINITY;

            for k in 0..num_neighbors {
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    dot += pout_slice[d] * routing_table[d * num_neighbors + k];
                }
                let l = dot * inv_temp;
                logits[k] = l;
                if l > max_logit {
                    max_logit = l;
                }
            }

            let mut sum_exp = 0.0f32;
            let mut exps = vec![0.0f32; num_neighbors];
            for k in 0..num_neighbors {
                let e = (logits[k] - max_logit).exp();
                exps[k] = e;
                sum_exp += e;
            }

            let inv_sum = 1.0f32 / (sum_exp + 1e-8f32);
            let mut probs = vec![0.0f32; num_neighbors];
            let mut entropy = 0.0f32;
            let mut best_gumbel = f32::NEG_INFINITY;
            let mut best_k = 0;

            for k in 0..num_neighbors {
                let raw_prob = exps[k] * inv_sum;
                let prob_val = 0.95f32 * raw_prob + noise_base;
                probs[k] = prob_val;

                if prob_val > 1e-10f32 {
                    entropy -= prob_val * (prob_val + 1e-10f32).log2();
                }

                let g_val = (prob_val + 1e-10f32).ln() + gnoise_slice[k];
                if g_val > best_gumbel {
                    best_gumbel = g_val;
                    best_k = k;
                }
            }

            chosen_neighbor[b] = best_k;

            // 3. Halting logic
            let hdr = headers[b];
            let mut is_halted = hdr.is_halted();
            if !is_halted && hdr.hop_count >= min_hop && delta_p < epsilon_p && entropy < epsilon_h
            {
                is_halted = true;
            }
            halting_flags[b] = is_halted;
        }
    }
}

#[derive(Debug)]
pub struct CudaDeviceWeights {
    pub d_w_gate: *mut f32,
    pub d_w_up: *mut f32,
    pub d_w_down: *mut f32,
}

impl CudaDeviceWeights {
    #[allow(unused_variables)]
    pub fn new(w_gate: &[f32], w_up: &[f32], w_down: &[f32]) -> Self {
        #[cfg(cuda_available)]
        {
            let mut d_w_gate: *mut f32 = std::ptr::null_mut();
            let mut d_w_up: *mut f32 = std::ptr::null_mut();
            let mut d_w_down: *mut f32 = std::ptr::null_mut();

            unsafe {
                if cudaMalloc(
                    &mut d_w_gate as *mut _ as *mut *mut std::ffi::c_void,
                    w_gate.len() * 4,
                ) == 0
                {
                    cudaMemcpy(
                        d_w_gate as *mut std::ffi::c_void,
                        w_gate.as_ptr() as *const std::ffi::c_void,
                        w_gate.len() * 4,
                        1,
                    );
                }
                if cudaMalloc(
                    &mut d_w_up as *mut _ as *mut *mut std::ffi::c_void,
                    w_up.len() * 4,
                ) == 0
                {
                    cudaMemcpy(
                        d_w_up as *mut std::ffi::c_void,
                        w_up.as_ptr() as *const std::ffi::c_void,
                        w_up.len() * 4,
                        1,
                    );
                }
                if cudaMalloc(
                    &mut d_w_down as *mut _ as *mut *mut std::ffi::c_void,
                    w_down.len() * 4,
                ) == 0
                {
                    cudaMemcpy(
                        d_w_down as *mut std::ffi::c_void,
                        w_down.as_ptr() as *const std::ffi::c_void,
                        w_down.len() * 4,
                        1,
                    );
                }
            }

            Self {
                d_w_gate,
                d_w_up,
                d_w_down,
            }
        }
        #[cfg(not(cuda_available))]
        {
            Self {
                d_w_gate: std::ptr::null_mut(),
                d_w_up: std::ptr::null_mut(),
                d_w_down: std::ptr::null_mut(),
            }
        }
    }
}

impl Drop for CudaDeviceWeights {
    fn drop(&mut self) {
        #[cfg(cuda_available)]
        unsafe {
            if !self.d_w_gate.is_null() {
                cudaFree(self.d_w_gate as *mut std::ffi::c_void);
            }
            if !self.d_w_up.is_null() {
                cudaFree(self.d_w_up as *mut std::ffi::c_void);
            }
            if !self.d_w_down.is_null() {
                cudaFree(self.d_w_down as *mut std::ffi::c_void);
            }
        }
    }
}

unsafe impl Send for CudaDeviceWeights {}
unsafe impl Sync for CudaDeviceWeights {}
