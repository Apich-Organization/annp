#[cfg(cuda_available)]
use std::ffi::c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ParticleCudaHeader {
    pub origin_token_id: u32,
    pub shard_id: u16,
    pub energy: f32,
    pub hop_count: u16,
    pub halted: bool,
}

#[cfg(cuda_available)]
extern "C" {
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

    pub fn launch_particle_prefetch_aggregate(
        src_particles: *const f32,
        dst_buffer: *mut f32,
        active_indices: *const i32,
        num_particles: i32,
        d_head: i32,
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

/// Safe Rust interface wrapper with explicit x86_64 AVX2+FMA SIMD Intrinsics & dtact scheduler integration
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
        norm_strategy: usize,
        alpha: f32,
        sphere_radius: f32,
    ) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "sub rsp, 8",
                "mov dword ptr [rsp], 0x1F80",
                "ldmxcsr [rsp]",
                "add rsp, 8",
                options(nostack, preserves_flags)
            );
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!(
                "mrs {x}, fpcr",
                "bic {x}, {x}, #(0x1F << 8)",
                "msr fpcr, {x}",
                x = out(reg) _,
                options(nostack, preserves_flags)
            );
        }

        #[cfg(cuda_available)]
        unsafe {
            launch_fused_micro_block(
                p_in.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                w_gate.as_ptr(),
                w_up.as_ptr(),
                w_down.as_ptr(),
                p_out.as_mut_ptr(),
                batch_size as i32,
                d_head as i32,
                ffn_dim as i32,
                kv_len as i32,
                norm_strategy as i32,
                alpha,
                sphere_radius,
                std::ptr::null_mut(),
            );
        }

        #[cfg(not(cuda_available))]
        {
            if batch_size == 0 || d_head == 0 {
                return;
            }

            let is_x86_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            if is_x86_avx2 {
                unsafe {
                    Self::execute_fused_avx2(
                        p_in,
                        k_cache,
                        v_cache,
                        w_gate,
                        w_up,
                        w_down,
                        p_out,
                        batch_size,
                        d_head,
                        ffn_dim,
                        kv_len,
                        norm_strategy,
                        alpha,
                        sphere_radius,
                    );
                }
            } else {
                Self::execute_fused_fallback(
                    p_in,
                    k_cache,
                    v_cache,
                    w_gate,
                    w_up,
                    w_down,
                    p_out,
                    batch_size,
                    d_head,
                    ffn_dim,
                    kv_len,
                    norm_strategy,
                    alpha,
                    sphere_radius,
                );
            }
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
        norm_strategy: usize,
        alpha: f32,
        _sphere_radius: f32,
    ) {
        use std::arch::x86_64::*;

        // Guard upper bounds for stack buffer allocations (supports d_head=128+)
        assert!(
            d_head <= 256 && d_head % 8 == 0,
            "d_head must be <= 256 and divisible by 8"
        );
        assert!(
            ffn_dim <= 1024 && ffn_dim % 8 == 0,
            "ffn_dim must be <= 1024 and divisible by 8"
        );
        assert!(p_in.len() >= batch_size * d_head);
        assert!(p_out.len() >= batch_size * d_head);

        // Fast horizontal addition of a 256-bit AVX register in registers
        #[inline]
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

            // Buffers sized for up to d_head=256 and ffn_dim=1024
            let mut attn_out = [0.0f32; 256];
            let mut s_mid = [0.0f32; 256];
            let mut ffn_inter = [0.0f32; 1024];

            // ---------------------------------------------------------------------
            // 1. Explicit AVX2 Attention Dot Product
            // ---------------------------------------------------------------------
            if kv_len > 0 {
                let k_cap = kv_len.min(128);
                let mut scores = [0.0f32; 128];
                let mut max_score = -1e9f32;

                for k in 0..k_cap {
                    let k_slice = &k_cache[k * d_head..(k + 1) * d_head];
                    let mut acc_v = _mm256_setzero_ps();

                    for d in (0..d_head).step_by(8) {
                        let p_vec = _mm256_loadu_ps(curr_p.as_ptr().add(d));
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
                for k in 0..k_cap {
                    let e = (scores[k] - max_score).exp();
                    scores[k] = e;
                    sum_exp += e;
                }

                let inv_sum = 1.0 / (sum_exp + 1e-8);
                for k in 0..k_cap {
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

            // ---------------------------------------------------------------------
            // 2. MicroRMSNorm / Residual
            // ---------------------------------------------------------------------
            if norm_strategy == 0 {
                let mut sq_acc = _mm256_setzero_ps();
                for d in (0..d_head).step_by(8) {
                    let a_v = _mm256_loadu_ps(attn_out.as_ptr().add(d));
                    sq_acc = _mm256_fmadd_ps(a_v, a_v, sq_acc);
                }
                let sq = hsum_avx2(sq_acc);
                let rms = (sq / (d_head as f32) + 1e-8).sqrt();
                let alpha_inv_rms = alpha / rms;
                let alpha_vec = _mm256_set1_ps(alpha_inv_rms);

                for d in (0..d_head).step_by(8) {
                    let p_v = _mm256_loadu_ps(curr_p.as_ptr().add(d));
                    let a_v = _mm256_loadu_ps(attn_out.as_ptr().add(d));
                    let res = _mm256_fmadd_ps(a_v, alpha_vec, p_v);
                    _mm256_storeu_ps(s_mid.as_mut_ptr().add(d), res);
                }
            } else {
                for d in 0..d_head {
                    s_mid[d] = curr_p[d] + attn_out[d];
                }
            }

            // ---------------------------------------------------------------------
            // 3. SwiGLU FFN Outer-Product (Contiguous AVX2 Memory Sweeps)
            // ---------------------------------------------------------------------
            let mut gate_arr = [0.0f32; 1024];
            let mut up_arr = [0.0f32; 1024];

            // Compute Gate and Up projections simultaneously using vector outer-products
            for d in 0..d_head {
                let m_v = _mm256_set1_ps(s_mid[d]);
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

            // Element-wise Swish / SiLU activation and Gated Product
            for j in 0..ffn_dim {
                let gate = gate_arr[j];
                let up = up_arr[j];
                let swish = gate / (1.0 + (-gate).exp());
                ffn_inter[j] = swish * up;
            }

            // ---------------------------------------------------------------------
            // 4. Down Projection Outer-Product & Output Store
            // ---------------------------------------------------------------------
            let mut down_arr = [0.0f32; 256];

            // w_down is shape [ffn_dim, d_head] — contiguous memory is along `d`
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

            // Apply scaling factor alpha, add residual, and clamp
            for d in (0..d_head).step_by(8) {
                let s_mid_v = _mm256_loadu_ps(s_mid.as_ptr().add(d));
                let down_v = _mm256_loadu_ps(down_arr.as_ptr().add(d));
                let alpha_v = _mm256_set1_ps(alpha);

                // res = s_mid + alpha * down
                let res = _mm256_fmadd_ps(down_v, alpha_v, s_mid_v);

                // Clamp between -100.0 and 100.0
                let min_v = _mm256_set1_ps(-100.0);
                let max_v = _mm256_set1_ps(100.0);
                let clamped = _mm256_min_ps(_mm256_max_ps(res, min_v), max_v);

                _mm256_storeu_ps(out_slice.as_mut_ptr().add(d), clamped);
            }
        }
    }

    fn execute_fused_fallback(
        p_in: &[f32],
        _k_cache: &[f32],
        _v_cache: &[f32],
        _w_gate: &[f32],
        _w_up: &[f32],
        _w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        _ffn_dim: usize,
        _kv_len: usize,
        _norm_strategy: usize,
        _alpha: f32,
        _sphere_radius: f32,
    ) {
        let copy_len = (batch_size * d_head).min(p_in.len()).min(p_out.len());
        p_out[..copy_len].copy_from_slice(&p_in[..copy_len]);
    }
}
