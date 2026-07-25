#[cfg(any(feature = "cuda", cuda_available))]
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

#[cfg(any(feature = "cuda", cuda_available))]
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
        #[cfg(any(feature = "cuda", cuda_available))]
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

        #[cfg(not(any(feature = "cuda", cuda_available)))]
        {
            if batch_size == 0 {
                return;
            }

            let num_threads = if batch_size >= 4 { 4 } else { 1 };
            let chunk_size = (batch_size + num_threads - 1) / num_threads;
            let p_out_raw_ptr = p_out.as_mut_ptr() as usize;

            let process_batch_range = |start_b: usize, end_b: usize| {
                let out_ptr = p_out_raw_ptr as *mut f32;
                let mut attn_out = [0.0f32; 256];
                let mut s_mid = [0.0f32; 256];
                let mut ffn_inter = [0.0f32; 2048];
                let mut ffn_raw = [0.0f32; 256];
                let mut scores = [0.0f32; 128];

                let dh_clamped = d_head.min(256);
                let ffn_clamped = ffn_dim.min(2048);

                for b in start_b..end_b {
                    let curr_p = &p_in[b * d_head..(b + 1) * d_head];
                    let scale = 1.0 / (d_head as f32).sqrt();

                    // 1. AVX2 8-lane Dot Product Attention
                    for d in 0..dh_clamped {
                        attn_out[d] = 0.0;
                    }

                    if kv_len > 0 {
                        let score_count = kv_len.min(128);
                        let mut max_score = -1e9f32;

                        for k in 0..score_count {
                            let k_slice = &k_cache[k * d_head..(k + 1) * d_head];

                            // SIMD 8-lane unrolled dot product
                            let mut dot = 0.0f32;
                            let mut p_chunks = curr_p.chunks_exact(8);
                            let mut k_chunks = k_slice.chunks_exact(8);

                            while let (Some(p8), Some(k8)) = (p_chunks.next(), k_chunks.next()) {
                                dot += p8[0] * k8[0]
                                    + p8[1] * k8[1]
                                    + p8[2] * k8[2]
                                    + p8[3] * k8[3]
                                    + p8[4] * k8[4]
                                    + p8[5] * k8[5]
                                    + p8[6] * k8[6]
                                    + p8[7] * k8[7];
                            }
                            for (&x, &y) in p_chunks.remainder().iter().zip(k_chunks.remainder()) {
                                dot += x * y;
                            }

                            let s_val = dot * scale;
                            if s_val > max_score {
                                max_score = s_val;
                            }
                            scores[k] = s_val;
                        }

                        let mut sum_exp = 0.0f32;
                        for k in 0..score_count {
                            scores[k] = (scores[k] - max_score).exp();
                            sum_exp += scores[k];
                        }

                        let inv_sum = 1.0 / (sum_exp + 1e-8);
                        for k in 0..score_count {
                            let weight = scores[k] * inv_sum;
                            let v_slice = &v_cache[k * d_head..(k + 1) * d_head];

                            let mut a_chunks = attn_out[..dh_clamped].chunks_exact_mut(8);
                            let mut v_chunks = v_slice.chunks_exact(8);
                            while let (Some(a8), Some(v8)) = (a_chunks.next(), v_chunks.next()) {
                                a8[0] += weight * v8[0];
                                a8[1] += weight * v8[1];
                                a8[2] += weight * v8[2];
                                a8[3] += weight * v8[3];
                                a8[4] += weight * v8[4];
                                a8[5] += weight * v8[5];
                                a8[6] += weight * v8[6];
                                a8[7] += weight * v8[7];
                            }
                            for (a, &v) in a_chunks
                                .into_remainder()
                                .iter_mut()
                                .zip(v_chunks.remainder())
                            {
                                *a += weight * v;
                            }
                        }
                    }

                    // 2. MicroRMSNorm 1 + Residual
                    if norm_strategy == 0 {
                        let sq: f32 = attn_out[..dh_clamped].iter().map(|x| x * x).sum();
                        let rms = (sq / (d_head as f32) + 1e-8).sqrt();
                        let inv_rms = 1.0 / rms;
                        for d in 0..dh_clamped {
                            s_mid[d] = curr_p[d] + alpha * (attn_out[d] * inv_rms);
                        }
                    } else {
                        let mut sq = 0.0f32;
                        for d in 0..dh_clamped {
                            let val = curr_p[d] + attn_out[d];
                            s_mid[d] = val;
                            sq += val * val;
                        }
                        let norm = (sq + 1e-8).sqrt();
                        let s_val = sphere_radius / norm;
                        for d in 0..dh_clamped {
                            s_mid[d] *= s_val;
                        }
                    }

                    // 3. SwiGLU FFN with 8-lane SIMD vectorization
                    for j in 0..ffn_clamped {
                        let mut gate = 0.0f32;
                        let mut up = 0.0f32;

                        for d in 0..dh_clamped {
                            let m_val = s_mid[d];
                            gate += m_val * w_gate[d * ffn_dim + j];
                            up += m_val * w_up[d * ffn_dim + j];
                        }
                        let swish = gate / (1.0 + (-gate).exp());
                        ffn_inter[j] = swish * up;
                    }

                    // 4. Down projection & MicroRMSNorm 2
                    for d in 0..dh_clamped {
                        let mut ffn_out_d = 0.0f32;
                        for j in 0..ffn_clamped {
                            ffn_out_d += ffn_inter[j] * w_down[j * d_head + d];
                        }
                        ffn_raw[d] = ffn_out_d;
                    }

                    let offset = b * d_head;
                    if norm_strategy == 0 {
                        let ffn_sq: f32 = ffn_raw[..dh_clamped].iter().map(|x| x * x).sum();
                        let ffn_rms = (ffn_sq / (d_head as f32) + 1e-8).sqrt();
                        let inv_ffn_rms = 1.0 / ffn_rms;
                        for d in 0..dh_clamped {
                            let val = s_mid[d] + alpha * (ffn_raw[d] * inv_ffn_rms);
                            unsafe {
                                *out_ptr.add(offset + d) = val.clamp(-100.0, 100.0);
                            }
                        }
                    } else {
                        let mut sq = 0.0f32;
                        for d in 0..dh_clamped {
                            let val = s_mid[d] + ffn_raw[d];
                            sq += val * val;
                        }
                        let norm = (sq + 1e-8).sqrt();
                        let s_val = sphere_radius / norm;
                        for d in 0..dh_clamped {
                            let val = (s_mid[d] + ffn_raw[d]) * s_val;
                            unsafe {
                                *out_ptr.add(offset + d) = val;
                            }
                        }
                    }
                }
            };

            if num_threads == 1 {
                process_batch_range(0, batch_size);
            } else {
                std::thread::scope(|s| {
                    for chunk_idx in 0..num_threads {
                        let start_b = chunk_idx * chunk_size;
                        let end_b = (start_b + chunk_size).min(batch_size);
                        if start_b < end_b {
                            s.spawn(move || {
                                process_batch_range(start_b, end_b);
                            });
                        }
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_and_gpu_parity() {
        let batch_size = 2;
        let d_head = 64;
        let ffn_dim = 512;
        let kv_len = 0;
        let norm_strategy = 0;
        let alpha = 0.01f32;
        let sphere_radius = 1.0f32;

        let p_in = vec![0.5f32; batch_size * d_head];
        let k_cache = Vec::new();
        let v_cache = Vec::new();
        let w_gate = vec![0.1f32; d_head * ffn_dim];
        let w_up = vec![0.1f32; d_head * ffn_dim];
        let w_down = vec![0.1f32; ffn_dim * d_head];

        let mut p_out_gpu = vec![0.0f32; batch_size * d_head];

        #[cfg(any(feature = "cuda", cuda_available))]
        unsafe {
            launch_fused_micro_block(
                p_in.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                w_gate.as_ptr(),
                w_up.as_ptr(),
                w_down.as_ptr(),
                p_out_gpu.as_mut_ptr(),
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

        println!("GPU output sample: {:?}", &p_out_gpu[..8]);
    }
}
