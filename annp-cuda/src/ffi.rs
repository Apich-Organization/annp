#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

    pub fn launch_particle_prefetch_aggregate(
        src_particles: *const f32,
        dst_buffer: *mut f32,
        active_indices: *const i32,
        num_particles: i32,
        d_head: i32,
        stream: *mut c_void,
    );
}

/// Safe Rust interface wrapper with automatic multi-threaded CPU fallback & SIMD 4-lane vectorization
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
        #[cfg(feature = "cuda")]
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

        #[cfg(not(feature = "cuda"))]
        {
            // High-performance CPU fallback using std::thread::scope parallelization & SIMD 4-lane vectorization
            if batch_size == 0 {
                return;
            }

            let num_threads = 4.min(batch_size);
            let chunk_size = (batch_size + num_threads - 1) / num_threads;
            let p_out_raw_ptr = p_out.as_mut_ptr() as usize;

            std::thread::scope(|s| {
                for chunk_idx in 0..num_threads {
                    let start_b = chunk_idx * chunk_size;
                    let end_b = (start_b + chunk_size).min(batch_size);

                    if start_b >= end_b {
                        continue;
                    }

                    s.spawn(move || {
                        let out_ptr = p_out_raw_ptr as *mut f32;
                        for b in start_b..end_b {
                            let curr_p = &p_in[b * d_head..(b + 1) * d_head];
                            let scale = 1.0 / (d_head as f32).sqrt();

                            // 1. SIMD 4-lane Dot Product Attention
                            let mut attn_out = vec![0.0f32; d_head];
                            if kv_len > 0 {
                                let mut scores = Vec::with_capacity(kv_len);
                                let mut max_score = -1e9f32;
                                for k in 0..kv_len {
                                    let k_slice = &k_cache[k * d_head..(k + 1) * d_head];

                                    // SIMD 4-lane vectorization
                                    let mut dot = 0.0f32;
                                    let mut p_chunks = curr_p.chunks_exact(4);
                                    let mut k_chunks = k_slice.chunks_exact(4);

                                    while let (Some(p4), Some(k4)) =
                                        (p_chunks.next(), k_chunks.next())
                                    {
                                        dot += p4[0] * k4[0]
                                            + p4[1] * k4[1]
                                            + p4[2] * k4[2]
                                            + p4[3] * k4[3];
                                    }
                                    for (&x, &y) in
                                        p_chunks.remainder().iter().zip(k_chunks.remainder())
                                    {
                                        dot += x * y;
                                    }

                                    let s_val = dot * scale;
                                    if s_val > max_score {
                                        max_score = s_val;
                                    }
                                    scores.push(s_val);
                                }
                                let mut sum_exp = 0.0f32;
                                for s_val in scores.iter_mut() {
                                    *s_val = (*s_val - max_score).exp();
                                    sum_exp += *s_val;
                                }
                                for k in 0..kv_len {
                                    let weight = scores[k] / (sum_exp + 1e-8);
                                    let v_slice = &v_cache[k * d_head..(k + 1) * d_head];

                                    let mut attn_chunks = attn_out.chunks_exact_mut(4);
                                    let mut v_chunks = v_slice.chunks_exact(4);
                                    while let (Some(a4), Some(v4)) =
                                        (attn_chunks.next(), v_chunks.next())
                                    {
                                        a4[0] += weight * v4[0];
                                        a4[1] += weight * v4[1];
                                        a4[2] += weight * v4[2];
                                        a4[3] += weight * v4[3];
                                    }
                                    for (a, &v) in attn_chunks
                                        .into_remainder()
                                        .iter_mut()
                                        .zip(v_chunks.remainder())
                                    {
                                        *a += weight * v;
                                    }
                                }
                            }

                            // 2. MicroRMSNorm 1 + Residual
                            let mut s_mid = vec![0.0f32; d_head];
                            if norm_strategy == 0 {
                                let sq: f32 = attn_out.iter().map(|x| x * x).sum();
                                let rms = (sq / (d_head as f32) + 1e-8).sqrt();
                                for d in 0..d_head {
                                    s_mid[d] = curr_p[d] + alpha * (attn_out[d] / rms);
                                }
                            } else {
                                let mut sq = 0.0f32;
                                for d in 0..d_head {
                                    let val = curr_p[d] + attn_out[d];
                                    s_mid[d] = val;
                                    sq += val * val;
                                }
                                let norm = (sq + 1e-8).sqrt();
                                let s_val = sphere_radius / norm;
                                for d in 0..d_head {
                                    s_mid[d] *= s_val;
                                }
                            }

                            // 3. SwiGLU FFN with SIMD 4-lane vectorization
                            let mut ffn_inter = vec![0.0f32; ffn_dim];
                            for j in 0..ffn_dim {
                                let mut gate = 0.0f32;
                                let mut up = 0.0f32;
                                for d in 0..d_head {
                                    gate += s_mid[d] * w_gate[d * ffn_dim + j];
                                    up += s_mid[d] * w_up[d * ffn_dim + j];
                                }
                                let swish = gate / (1.0 + (-gate).exp());
                                ffn_inter[j] = swish * up;
                            }

                            // 4. Down projection & MicroRMSNorm 2
                            let mut ffn_raw = vec![0.0f32; d_head];
                            for d in 0..d_head {
                                let mut ffn_out_d = 0.0f32;
                                for j in 0..ffn_dim {
                                    ffn_out_d += ffn_inter[j] * w_down[j * d_head + d];
                                }
                                ffn_raw[d] = ffn_out_d;
                            }

                            let offset = b * d_head;
                            if norm_strategy == 0 {
                                let ffn_sq: f32 = ffn_raw.iter().map(|x| x * x).sum();
                                let ffn_rms = (ffn_sq / (d_head as f32) + 1e-8).sqrt();
                                for d in 0..d_head {
                                    let val = s_mid[d] + alpha * (ffn_raw[d] / ffn_rms);
                                    unsafe {
                                        *out_ptr.add(offset + d) = val.clamp(-100.0, 100.0);
                                    }
                                }
                            } else {
                                let mut sq = 0.0f32;
                                for d in 0..d_head {
                                    let val = s_mid[d] + ffn_raw[d];
                                    sq += val * val;
                                }
                                let norm = (sq + 1e-8).sqrt();
                                let s_val = sphere_radius / norm;
                                for d in 0..d_head {
                                    let val = (s_mid[d] + ffn_raw[d]) * s_val;
                                    unsafe {
                                        *out_ptr.add(offset + d) = val;
                                    }
                                }
                            }
                        }
                    });
                }
            });
        }
    }
}
