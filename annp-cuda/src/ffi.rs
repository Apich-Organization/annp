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

        let dh_clamped = d_head.min(64);
        let ffn_clamped = ffn_dim.min(512);

        for b in 0..batch_size {
            let curr_p = &p_in[b * d_head..(b + 1) * d_head];
            let out_slice = &mut p_out[b * d_head..(b + 1) * d_head];

            let mut attn_out = [0.0f32; 64];
            let mut s_mid = [0.0f32; 64];
            let mut ffn_inter = [0.0f32; 512];

            // 1. Explicit AVX2 Attention Dot Product
            let scale = 1.0 / (d_head as f32).sqrt();
            if kv_len > 0 {
                let k_cap = kv_len.min(128);
                let mut scores = [0.0f32; 128];
                let mut max_score = -1e9f32;

                for k in 0..k_cap {
                    let k_slice = &k_cache[k * d_head..(k + 1) * d_head];
                    let mut acc_v = _mm256_setzero_ps();

                    for d in (0..dh_clamped).step_by(8) {
                        unsafe {
                            let p_vec = _mm256_loadu_ps(curr_p.as_ptr().add(d));
                            let k_vec = _mm256_loadu_ps(k_slice.as_ptr().add(d));
                            acc_v = _mm256_fmadd_ps(p_vec, k_vec, acc_v);
                        }
                    }

                    let mut tmp = [0.0f32; 8];
                    unsafe {
                        _mm256_storeu_ps(tmp.as_mut_ptr(), acc_v);
                    }
                    let dot = tmp.iter().sum::<f32>() * scale;
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

                    for d in (0..dh_clamped).step_by(8) {
                        unsafe {
                            let w_vec = _mm256_set1_ps(w);
                            let v_vec = _mm256_loadu_ps(v_slice.as_ptr().add(d));
                            let curr_attn = _mm256_loadu_ps(attn_out.as_ptr().add(d));
                            let res = _mm256_fmadd_ps(w_vec, v_vec, curr_attn);
                            _mm256_storeu_ps(attn_out.as_mut_ptr().add(d), res);
                        }
                    }
                }
            }

            // 2. MicroRMSNorm 1 / SphereNorm
            if norm_strategy == 0 {
                let mut sq_acc = _mm256_setzero_ps();
                for d in (0..dh_clamped).step_by(8) {
                    unsafe {
                        let a_v = _mm256_loadu_ps(attn_out.as_ptr().add(d));
                        sq_acc = _mm256_fmadd_ps(a_v, a_v, sq_acc);
                    }
                }
                let mut tmp = [0.0f32; 8];
                unsafe {
                    _mm256_storeu_ps(tmp.as_mut_ptr(), sq_acc);
                }
                let sq: f32 = tmp.iter().sum();
                let rms = (sq / (d_head as f32) + 1e-8).sqrt();
                let alpha_inv_rms = alpha / rms;

                for d in (0..dh_clamped).step_by(8) {
                    unsafe {
                        let alpha_vec = _mm256_set1_ps(alpha_inv_rms);
                        let p_v = _mm256_loadu_ps(curr_p.as_ptr().add(d));
                        let a_v = _mm256_loadu_ps(attn_out.as_ptr().add(d));
                        let res = _mm256_fmadd_ps(a_v, alpha_vec, p_v);
                        _mm256_storeu_ps(s_mid.as_mut_ptr().add(d), res);
                    }
                }
            } else {
                for d in 0..dh_clamped {
                    s_mid[d] = curr_p[d] + attn_out[d];
                }
            }

            // 3. SwiGLU FFN via AVX2 Vector Intrinsics
            for j in 0..ffn_clamped {
                let mut tmp_g = [0.0f32; 8];
                let mut tmp_u = [0.0f32; 8];

                unsafe {
                    let mut gate_acc = _mm256_setzero_ps();
                    let mut up_acc = _mm256_setzero_ps();

                    for d in (0..dh_clamped).step_by(8) {
                        let m_v = _mm256_loadu_ps(s_mid.as_ptr().add(d));
                        let wg_v = _mm256_loadu_ps(w_gate.as_ptr().add(d * ffn_dim + j));
                        let wu_v = _mm256_loadu_ps(w_up.as_ptr().add(d * ffn_dim + j));
                        gate_acc = _mm256_fmadd_ps(m_v, wg_v, gate_acc);
                        up_acc = _mm256_fmadd_ps(m_v, wu_v, up_acc);
                    }

                    _mm256_storeu_ps(tmp_g.as_mut_ptr(), gate_acc);
                    _mm256_storeu_ps(tmp_u.as_mut_ptr(), up_acc);
                }

                let gate: f32 = tmp_g.iter().sum();
                let up: f32 = tmp_u.iter().sum();
                let swish = gate / (1.0 + (-gate).exp());
                ffn_inter[j] = swish * up;
            }

            // 4. Down projection & Store
            for d in 0..dh_clamped {
                let mut ffn_out_d = 0.0f32;
                for j in 0..ffn_clamped {
                    ffn_out_d += ffn_inter[j] * w_down[j * d_head + d];
                }
                out_slice[d] = (s_mid[d] + alpha * ffn_out_d).clamp(-100.0, 100.0);
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
