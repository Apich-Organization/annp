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

/// Safe Rust interface wrapper with automatic CPU fallback when compiled without CUDA feature
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
            // Fully functional CPU fallback implementation matching CUDA kernel math 1:1
            for b in 0..batch_size {
                let curr_p = &p_in[b * d_head..(b + 1) * d_head];
                let curr_out = &mut p_out[b * d_head..(b + 1) * d_head];
                let scale = 1.0 / (d_head as f32).sqrt();

                // 1. Attention
                let mut attn_out = vec![0.0f32; d_head];
                if kv_len > 0 {
                    let mut scores = Vec::with_capacity(kv_len);
                    let mut max_score = -1e9f32;
                    for k in 0..kv_len {
                        let k_slice = &k_cache[k * d_head..(k + 1) * d_head];
                        let dot: f32 = curr_p.iter().zip(k_slice.iter()).map(|(x, y)| x * y).sum();
                        let s = dot * scale;
                        if s > max_score {
                            max_score = s;
                        }
                        scores.push(s);
                    }
                    let mut sum_exp = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max_score).exp();
                        sum_exp += *s;
                    }
                    for k in 0..kv_len {
                        let weight = scores[k] / (sum_exp + 1e-8);
                        let v_slice = &v_cache[k * d_head..(k + 1) * d_head];
                        for d in 0..d_head {
                            attn_out[d] += weight * v_slice[d];
                        }
                    }
                }

                // 2. Norm 1 + Residual
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
                    let s = sphere_radius / norm;
                    for d in 0..d_head {
                        s_mid[d] *= s;
                    }
                }

                // 3. SwiGLU FFN
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

                // 4. Down projection & Norm 2
                for d in 0..d_head {
                    let mut ffn_out = 0.0f32;
                    for j in 0..ffn_dim {
                        ffn_out += ffn_inter[j] * w_down[j * d_head + d];
                    }
                    curr_out[d] = s_mid[d] + alpha * ffn_out;
                }
            }
        }
    }
}
