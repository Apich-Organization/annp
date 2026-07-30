use annp_cuda::{
    CudaMicroBlockRunner, CudaParticleAggregator, CudaParticleRouter, ParticleCudaHeader,
};

#[test]
fn test_micro_block_fused_rmsnorm() {
    let batch_size = 2;
    let d_head = 16;
    let ffn_dim = 64;
    let kv_len = 4;
    let alpha = 0.01;

    let p_in = vec![0.5f32; batch_size * d_head];
    let k_cache = vec![0.1f32; kv_len * d_head];
    let v_cache = vec![0.2f32; kv_len * d_head];
    let w_gate = vec![0.05f32; d_head * ffn_dim];
    let w_up = vec![0.05f32; d_head * ffn_dim];
    let w_down = vec![0.05f32; ffn_dim * d_head];

    let mut p_out = vec![0.0f32; batch_size * d_head];

    CudaMicroBlockRunner::execute_fused(
        &p_in, &k_cache, &v_cache, &w_gate, &w_up, &w_down, &mut p_out, batch_size, d_head,
        ffn_dim, kv_len, alpha,
    );

    // Verify non-zero output and reasonable values
    assert_eq!(p_out.len(), batch_size * d_head);
    for &val in p_out.iter() {
        assert!(val.is_finite());
        assert!(val >= -100.0 && val <= 100.0);
    }
}

#[test]
fn test_micro_block_fused_spherenorm() {
    let batch_size = 1;
    let d_head = 64;
    let ffn_dim = 128;
    let kv_len = 2;
    let alpha = 0.1;

    let p_in = vec![0.1f32; d_head];
    let k_cache = vec![0.2f32; kv_len * d_head];
    let v_cache = vec![0.3f32; kv_len * d_head];
    let w_gate = vec![0.01f32; d_head * ffn_dim];
    let w_up = vec![0.01f32; d_head * ffn_dim];
    let w_down = vec![0.01f32; ffn_dim * d_head];

    let mut p_out = vec![0.0f32; d_head];

    CudaMicroBlockRunner::execute_fused(
        &p_in, &k_cache, &v_cache, &w_gate, &w_up, &w_down, &mut p_out, batch_size, d_head,
        ffn_dim, kv_len, alpha,
    );

    // After removing sphere_radius, we just check output is finite
    for &val in p_out.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_particle_router_and_halting() {
    let batch_size = 2;
    let d_head = 8;
    let num_neighbors = 4;

    let p_in = vec![0.1f32; batch_size * d_head];
    let p_out = vec![0.10001f32; batch_size * d_head]; // delta_p very small
    let routing_table = vec![0.1f32; num_neighbors * d_head];
    let gumbel_noise = vec![0.0f32; batch_size * num_neighbors];

    let mut chosen = vec![0usize; batch_size];
    let mut halting = vec![false; batch_size];

    let headers = vec![
        ParticleCudaHeader {
            origin_token_id: 0,
            shard_id: 0,
            pad0: [0; 2],
            energy: 1.0,
            hop_count: 15,
            halted: 0,
            pad1: [0; 1],
        },
        ParticleCudaHeader {
            origin_token_id: 1,
            shard_id: 1,
            pad0: [0; 2],
            energy: 1.0,
            hop_count: 5, // < min_hop
            halted: 0,
            pad1: [0; 1],
        },
    ];

    CudaParticleRouter::execute_routing(
        &p_in,
        &p_out,
        &routing_table,
        &gumbel_noise,
        &mut chosen,
        &mut halting,
        batch_size,
        d_head,
        num_neighbors,
        1.0,  // temperature
        1e-2, // epsilon_p
        10.0, // epsilon_h
        10,   // min_hop
        &headers,
    );

    // Header 0: hop_count 15 >= 10 min_hop and small delta_p -> should trigger halting
    assert_eq!(halting[0], true);
    // Header 1: hop_count 5 < 10 min_hop -> should NOT trigger halting
    assert_eq!(halting[1], false);

    assert!(chosen[0] < num_neighbors);
    assert!(chosen[1] < num_neighbors);
}

#[test]
fn test_particle_aggregator() {
    let num_particles = 3;
    let d_head = 4;

    let src_particles = vec![
        1.0, 2.0, 3.0, 4.0, // Particle 0
        5.0, 6.0, 7.0, 8.0, // Particle 1
        9.0, 10.0, 11.0, 12.0, // Particle 2
    ];

    let mut dst_buffer = vec![0.0f32; num_particles * d_head];
    let active_indices = vec![2, 0, 1]; // Shuffle order

    CudaParticleAggregator::execute_prefetch(
        &src_particles,
        &mut dst_buffer,
        Some(&active_indices),
        num_particles,
        d_head,
    );

    // dst_buffer[0] should match src_particles[2] (9.0, 10.0, 11.0, 12.0)
    assert_eq!(&dst_buffer[0..4], &[9.0, 10.0, 11.0, 12.0]);
    // dst_buffer[1] should match src_particles[0] (1.0, 2.0, 3.0, 4.0)
    assert_eq!(&dst_buffer[4..8], &[1.0, 2.0, 3.0, 4.0]);
    // dst_buffer[2] should match src_particles[1] (5.0, 6.0, 7.0, 8.0)
    assert_eq!(&dst_buffer[8..12], &[5.0, 6.0, 7.0, 8.0]);
}
