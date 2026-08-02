use crate::micro_block::MicroBlockNode;
use crate::subnode::Subnode;
use annp_core::MicroBlockConfig;

#[test]
fn test_subnode_initialization() {
    let d_head = 64;
    let ffn_dim = 256;
    let alpha = 1.0;

    let gamma = 0.99;
    let subnode = Subnode::new_random(0, d_head, ffn_dim, alpha, gamma);

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

    let gamma = 0.99;
    let parent = Subnode::new_random(0, d_head, ffn_dim, alpha, gamma);
    let child = Subnode::spawn_from_parent(1, &parent, d_head, ffn_dim, gamma, 1.0);

    assert_eq!(child.subnode_id, 1);

    // w_down should be slightly perturbed to prevent sudden drop in residual
    let all_zero = child.w_down.iter().all(|&w| w == 0.0);
    assert!(
        !all_zero,
        "Spawned subnode must have perturbed w_down to maintain smooth residual"
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

    let _max_kv_len = 128;

    let node = MicroBlockNode::new(0, config, false);

    assert_eq!(node.node_id, 0);
    assert_eq!(node.subnodes.len(), 1); // Only 1 primary subnode at start

    // Independent workspace buffer sizing
    assert_eq!(node.p_in_buf.len(), 0);
    assert_eq!(node.p_out_buf.len(), 0);
}
