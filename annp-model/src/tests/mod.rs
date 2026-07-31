use crate::micro_block::MicroBlockNode;
use crate::subnode::Subnode;
use annp_core::MicroBlockConfig;

#[test]
fn test_subnode_initialization() {
    let d_head = 64;
    let ffn_dim = 256;
    let alpha = 1.0;

    let subnode = Subnode::new_random(0, d_head, ffn_dim, alpha);

    assert_eq!(subnode.subnode_id, 0);
    assert_eq!(subnode.w_gate.len(), d_head * ffn_dim);
    assert_eq!(subnode.w_up.len(), d_head * ffn_dim);
    assert_eq!(subnode.w_down.len(), ffn_dim * d_head);

    // Check that down weights are initialized with a very small scale, not exact 0.0
    let has_non_zero = subnode.w_down.iter().any(|&w| w != 0.0);
    assert!(
        has_non_zero,
        "Randomly initialized w_down should not be all exactly zeros"
    );
}

#[test]
fn test_subnode_spawn_from_parent() {
    let d_head = 64;
    let ffn_dim = 256;
    let alpha = 1.0;

    let parent = Subnode::new_random(0, d_head, ffn_dim, alpha);
    let child = Subnode::spawn_from_parent(1, &parent, d_head, ffn_dim);

    assert_eq!(child.subnode_id, 1);

    // w_down should be fully zeroed on spawn to preserve identity transformation initially
    let all_zero = child.w_down.iter().all(|&w| w == 0.0);
    assert!(
        all_zero,
        "Spawned subnode must have zeroed w_down to maintain initial identity mapping"
    );

    // w_gate and w_up should be perturbed slightly but not identical
    let gate_differs = parent
        .w_gate
        .iter()
        .zip(child.w_gate.iter())
        .any(|(p, c)| p != c);
    assert!(gate_differs, "w_gate should have small perturbations");
}

#[test]
fn test_micro_block_node_creation() {
    let config = MicroBlockConfig::default();

    let max_kv_len = 128;

    let node = MicroBlockNode::new(0, config, max_kv_len, false);

    assert_eq!(node.node_id, 0);
    assert_eq!(node.subnodes.len(), 1); // Only 1 primary subnode at start
    assert_eq!(node.k_cache.capacity(), max_kv_len * 64);
    assert_eq!(node.v_cache.capacity(), max_kv_len * 64);

    // Independent workspace buffer sizing
    assert_eq!(node.p_in_buf.len(), 0);
    assert_eq!(node.p_out_buf.len(), 0);
}
