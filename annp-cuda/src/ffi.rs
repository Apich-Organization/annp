#[cfg(any(feature = "cuda", cuda_available))]
use std::ffi::c_void;

#[allow(unused_imports)]
use annp_core::RMS_EPSILON;

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

                    // === CPU fallback forward pass ===
                    // Must match micro_block.rs (inference path) and fused_micro_block_kernel exactly.
                    //
                    // Correct sequence:
                    //   1. p_in_normed  = RMSNorm(p_in)
                    //   2. attn_out     = fast_weight @ p_in_normed
                    //   3. s_mid        = p_in + alpha * attn_out        [raw residual, NO separate norm]
                    //   4. s_mid_normed = RMSNorm(s_mid)                 [norm the accumulated state]
                    //   5. gate_j, up_j = w_gate, w_up @ s_mid_normed
                    //   6. h_j          = swish(gate_j) * up_j  (SwiGLU)
                    //   7. ffn_raw      = w_down @ h
                    //   8. p_out        = s_mid + alpha * ffn_raw         [raw residual, NO separate norm]
                    //
                    // PREVIOUS BUG (fixed here):
                    //   Old code incorrectly applied RMSNorm to attn_out *before* the residual add:
                    //     s_mid = p_in + alpha * (attn_out / RMS(attn_out))   ← WRONG
                    //   and to ffn_raw before the second residual:
                    //     p_out = s_mid + alpha * (ffn_raw / RMS(ffn_raw))    ← WRONG
                    //   This is a completely different formula: it normalizes the *delta* rather than
                    //   the *accumulated state*, producing different outputs from both the CUDA kernel
                    //   and the canonical micro_block.rs path.

                    // Step 1: RMSNorm(p_in)
                    let sq_sum_in: f32 = curr_p.iter().map(|&v| v * v).sum();
                    let inv_rms_in = 1.0 / (sq_sum_in / (d_head as f32) + RMS_EPSILON).sqrt();
                    let mut p_in_normed = vec![0.0f32; d_head];
                    for d in 0..d_head {
                        p_in_normed[d] = curr_p[d] * inv_rms_in;
                    }

                    // Step 2: fast_weight @ p_in_normed
                    let mut attn_out = vec![0.0f32; d_head];
                    for r in 0..d_head {
                        let mut sum = 0.0f32;
                        for c in 0..d_head {
                            sum += fast_weight[r * d_head + c] * p_in_normed[c];
                        }
                        attn_out[r] = sum;
                    }

                    // Step 3: s_mid = p_in + alpha * attn_out  (raw, no separate normalization)
                    let mut s_mid = vec![0.0f32; d_head];
                    for d in 0..d_head {
                        s_mid[d] = curr_p[d] + alpha * attn_out[d];
                    }

                    // Step 4: RMSNorm(s_mid) → s_mid_normed (norm the accumulated state for FFN input)
                    let sq_sum_mid: f32 = s_mid.iter().map(|&v| v * v).sum();
                    let inv_rms_mid = 1.0 / (sq_sum_mid / (d_head as f32) + RMS_EPSILON).sqrt();
                    let mut s_mid_normed = vec![0.0f32; d_head];
                    for d in 0..d_head {
                        s_mid_normed[d] = s_mid[d] * inv_rms_mid;
                    }

                    // Steps 5–6: SwiGLU FFN on normalized s_mid
                    let mut ffn_inter = vec![0.0f32; ffn_dim];
                    for j in 0..ffn_dim {
                        let mut gate = 0.0f32;
                        let mut up = 0.0f32;
                        for d in 0..d_head {
                            gate += s_mid_normed[d] * w_gate[d * ffn_dim + j];
                            up += s_mid_normed[d] * w_up[d * ffn_dim + j];
                        }
                        let swish = gate / (1.0 + (-gate).exp()); // swish = gate * sigmoid(gate)
                        ffn_inter[j] = swish * up;
                    }

                    // Step 7: Down projection
                    let mut ffn_raw = vec![0.0f32; d_head];
                    for d in 0..d_head {
                        let mut acc = 0.0f32;
                        for j in 0..ffn_dim {
                            acc += ffn_inter[j] * w_down[j * d_head + d];
                        }
                        ffn_raw[d] = acc;
                    }

                    // Step 8: p_out = s_mid + alpha * ffn_raw  (raw, no separate normalization)
                    // Clip matches CUDA kernel numerical guard.
                    let offset = b * d_head;
                    for d in 0..d_head {
                        let val = s_mid[d] + alpha * ffn_raw[d];
                        unsafe {
                            *out_ptr.add(offset + d) = val.clamp(-100.0, 100.0);
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
        for (i, slot) in out_assignments.iter_mut().enumerate().take(batch_size) {
            *slot = (i % num_nodes) as i32;
        }
    }
}
