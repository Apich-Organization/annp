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

#[repr(C)]
#[derive(Debug)]
pub struct CudaStreamManager {
    _private: [u8; 0],
}

#[derive(Debug)]
pub struct CudaDeviceWeights {
    pub gate_ptr: *mut f32,
    pub up_ptr: *mut f32,
    pub down_ptr: *mut f32,
}

impl CudaDeviceWeights {
    pub fn new(_w_gate: &[f32], _w_up: &[f32], _w_down: &[f32]) -> Self {
        Self {
            gate_ptr: std::ptr::null_mut(),
            up_ptr: std::ptr::null_mut(),
            down_ptr: std::ptr::null_mut(),
        }
    }
}

#[cfg(any(feature = "cuda", cuda_available))]
unsafe extern "C" {
    pub fn launch_fused_micro_block(
        p_in: *const f32,
        fast_weight: *const f32,
        w_gate: *const f32,
        w_up: *const f32,
        w_down: *const f32,
        p_out: *mut f32,
        batch_size: i32,
        d_head: i32,
        ffn_dim: i32,
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

pub struct CudaMicroBlockRunner;

impl CudaMicroBlockRunner {
    pub fn execute_fused(
        p_in: &[f32],
        fast_weight: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        ffn_dim: usize,
        norm_strategy: usize,
        alpha: f32,
        sphere_radius: f32,
        _stream: Option<&CudaStreamManager>,
        _d_weights: Option<&CudaDeviceWeights>,
    ) {
        #[cfg(any(feature = "cuda", cuda_available))]
        unsafe {
            let stream_ptr = _stream.map_or(std::ptr::null_mut(), |s| s as *const _ as *mut c_void);
            launch_fused_micro_block(
                p_in.as_ptr(),
                fast_weight.as_ptr(),
                _d_weights.map_or(w_gate.as_ptr(), |d| d.gate_ptr as *const f32),
                _d_weights.map_or(w_up.as_ptr(), |d| d.up_ptr as *const f32),
                _d_weights.map_or(w_down.as_ptr(), |d| d.down_ptr as *const f32),
                p_out.as_mut_ptr(),
                batch_size as i32,
                d_head as i32,
                ffn_dim as i32,
                norm_strategy as i32,
                alpha,
                sphere_radius,
                stream_ptr,
            );
        }

        #[cfg(not(any(feature = "cuda", cuda_available)))]
        {
            if batch_size == 0 {
                return;
            }

            let num_threads = 4.min(batch_size);
            let chunk_size = (batch_size + num_threads - 1) / num_threads;
            let p_out_raw_ptr = p_out.as_mut_ptr() as usize;

            let process_batch_range = |start_b: usize, end_b: usize| {
                let out_ptr = p_out_raw_ptr as *mut f32;
                for b in start_b..end_b {
                    let curr_p = &p_in[b * d_head..(b + 1) * d_head];

                    // 1. Fast Weight Matrix Multiply
                    let mut attn_out = vec![0.0f32; d_head];

                    let mut sq_sum_in = 0.0f32;
                    for &val in curr_p.iter() {
                        sq_sum_in += val * val;
                    }
                    let inv_rms_in = 1.0 / (sq_sum_in / (d_head as f32) + 1e-8).sqrt();

                    let mut p_in_normed = vec![0.0f32; d_head];
                    for d in 0..d_head {
                        p_in_normed[d] = curr_p[d] * inv_rms_in;
                    }

                    for r in 0..d_head {
                        let mut sum = 0.0f32;
                        for c in 0..d_head {
                            sum += fast_weight[r * d_head + c] * p_in_normed[c];
                        }
                        attn_out[r] = sum;
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

    pub fn execute_fused_with_stream(
        p_in: &[f32],
        fast_weight: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        p_out: &mut [f32],
        batch_size: usize,
        d_head: usize,
        ffn_dim: usize,
        alpha: f32,
        _stream: Option<&CudaStreamManager>,
        _d_weights: Option<&CudaDeviceWeights>,
    ) {
        Self::execute_fused(
            p_in,
            fast_weight,
            w_gate,
            w_up,
            w_down,
            p_out,
            batch_size,
            d_head,
            ffn_dim,
            0,
            alpha,
            1.0,
            _stream,
            _d_weights,
        );
    }

    pub fn execute_backward(
        _p_in: &[f32],
        _fast_weight: &[f32],
        _w_gate: &mut [f32],
        _w_up: &mut [f32],
        _w_down: &mut [f32],
        _local_err: &[f32],
        _d_head: usize,
        _ffn_dim: usize,
        _alpha: f32,
        _lr: f32,
        _weight_decay: f32,
        _stream: Option<&CudaStreamManager>,
        _use_cuda: bool,
        _d_weights: Option<&mut CudaDeviceWeights>,
    ) {
        // CPU backward pass handles all weight updates directly in Rust.
        // If CUDA is needed in future, implement backwards pass here.
    }
}

pub struct CudaParticleRouter;
impl CudaParticleRouter {
    pub fn route_particles(
        _particles_in: &[f32],
        _node_positions: &[f32],
        batch_size: usize,
        _d_model: usize,
        num_nodes: usize,
        _radius: f32,
        out_assignments: &mut [i32],
    ) {
        for i in 0..batch_size {
            out_assignments[i] = (i % num_nodes) as i32;
        }
    }
}
