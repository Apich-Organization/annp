use crate::micro_block::MicroBlockNode;
use crate::scattering::TokenScattering;
use crate::serializer::EgressSerializer;
use crate::topology::TopologyGrid;
use annp_core::{MicroBlockConfig, Particle};
use candle_core::{Device, Result, Tensor};
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

static DTACT_INIT: Once = Once::new();

/// Initializes the global Dtact runtime exactly once per process.
pub fn init_dtact_runtime() {
    DTACT_INIT.call_once(|| {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let runtime = dtact::GLOBAL_RUNTIME.get_or_init(|| {
            let scheduler = dtact::dta_scheduler::DtaScheduler::new(
                workers,
                dtact::dta_scheduler::TopologyMode::P2PMesh,
            );
            let pool = dtact::memory_management::ContextPool::new(
                512,
                524_288,
                dtact::memory_management::SafetyLevel::Safety0,
                0,
            )
            .expect("dtact runtime init failed");
            dtact::Runtime {
                scheduler,
                pool,
                started: core::sync::atomic::AtomicBool::new(false),
                shutdown: core::sync::atomic::AtomicBool::new(false),
            }
        });
        runtime.start();
    });
}

/// Full ANNP Architecture Pipeline with dtact coroutine/work-stealing lock-free scheduling.
pub struct ANNPModel {
    pub config: MicroBlockConfig,
    pub num_nodes: usize,
    pub scattering: TokenScattering,
    pub nodes: Vec<MicroBlockNode>,
    pub topology: TopologyGrid,
    pub serializer: EgressSerializer,
    pub device: Device,
    // Reusable double-buffer queues for zero-alloc P2P particle routing
    pub node_queues: Vec<Vec<Particle>>,
    pub next_queues: Vec<Vec<Particle>>,
}

impl ANNPModel {
    pub fn new(
        num_nodes: usize,
        num_shards: usize,
        config: MicroBlockConfig,
        device: Device,
    ) -> Self {
        let use_cuda = matches!(device, Device::Cuda(_));
        Self::new_with_cuda(num_nodes, num_shards, config, device, use_cuda)
    }

    pub fn new_with_cuda(
        num_nodes: usize,
        num_shards: usize,
        config: MicroBlockConfig,
        device: Device,
        use_cuda: bool,
    ) -> Self {
        init_dtact_runtime();

        let d_head = config.d_head;
        let scattering = TokenScattering::new(num_shards, d_head, 0.1);
        let topology = TopologyGrid::new(num_nodes, d_head, 4);
        let serializer = EgressSerializer::new(d_head, num_shards);

        let nodes = (0..num_nodes)
            .map(|i| MicroBlockNode::new(i, config.clone(), 64, use_cuda))
            .collect();

        let node_queues = vec![Vec::with_capacity(64); num_nodes];
        let next_queues = vec![Vec::with_capacity(64); num_nodes];

        Self {
            config,
            num_nodes,
            scattering,
            nodes,
            topology,
            serializer,
            device,
            node_queues,
            next_queues,
        }
    }

    /// Forward pass through ANNP P2P Mesh with lock-free dtact coroutine mesh scheduling
    pub fn forward(&mut self, embeddings: &Tensor) -> Result<Tensor> {
        // Reset node KV Caches before each forward batch to prevent cross-batch historical state pollution
        for node in self.nodes.iter_mut() {
            node.k_cache.clear();
            node.v_cache.clear();
            node.last_p_in.iter_mut().for_each(|x| *x = 0.0);
        }

        let (seq_len, _) = embeddings.dims2()?;
        let initial_particles = self
            .scattering
            .scatter_embeddings(embeddings, &self.config)?;

        for q in self.node_queues.iter_mut() {
            q.clear();
        }
        for q in self.next_queues.iter_mut() {
            q.clear();
        }

        // Ingress distribution
        let ingress_nodes = &self.scattering.ingress_node_indices;
        for (idx, p) in initial_particles.into_iter().enumerate() {
            let target_ingress = ingress_nodes[idx % ingress_nodes.len()];
            self.node_queues[target_ingress].push(p);
        }

        let mut halted_particles: Vec<Particle> =
            Vec::with_capacity(seq_len * self.scattering.num_shards);
        let mut active_loop = true;
        let mut step = 0;
        let max_steps = self.config.max_hop as usize + 20;

        let mut curr_batches: Vec<Vec<Particle>> = vec![Vec::with_capacity(64); self.num_nodes];

        while active_loop && step < max_steps {
            step += 1;
            active_loop = false;

            for q in self.next_queues.iter_mut() {
                q.clear();
            }

            for node_id in 0..self.num_nodes {
                curr_batches[node_id].clear();
                std::mem::swap(&mut curr_batches[node_id], &mut self.node_queues[node_id]);
            }

            let use_cuda_mode = self.nodes.first().map_or(false, |n| n.use_cuda);

            if use_cuda_mode {
                // GPU Mode: Execute active nodes on CUDA GPU stream without CPU multi-threading contention
                for (node, batch) in self.nodes.iter_mut().zip(curr_batches.iter_mut()) {
                    if !batch.is_empty() {
                        node.process_batch(batch);
                    }
                }
            } else {
                // CPU Mode: Multi-threaded AVX2 SIMD execution via dtact worker pool
                let active_counter = AtomicUsize::new(0);
                let counter_ptr = &active_counter as *const AtomicUsize as usize;

                for (node, batch) in self.nodes.iter_mut().zip(curr_batches.iter_mut()) {
                    if !batch.is_empty() {
                        active_counter.fetch_add(1, Ordering::Relaxed);
                        let node_addr = node as *mut MicroBlockNode as usize;
                        let batch_ptr = batch as *mut Vec<Particle> as usize;

                        let _handle = dtact::spawn(async move {
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

                            let node_ptr = node_addr as *mut MicroBlockNode;
                            let b_ptr = batch_ptr as *mut Vec<Particle>;
                            let c_ptr = counter_ptr as *const AtomicUsize;
                            unsafe {
                                (*node_ptr).process_batch(&mut *b_ptr);
                                (*c_ptr).fetch_sub(1, Ordering::Release);
                            }
                        });
                    }
                }

                while active_counter.load(Ordering::Acquire) > 0 {
                    std::thread::yield_now();
                }
            }

            // Push-routing decision for output particles with P2P Backpressure penalty
            for node_id in 0..self.num_nodes {
                let curr_batch = &mut curr_batches[node_id];
                if curr_batch.is_empty() {
                    continue;
                }
                active_loop = true;

                let neighbors = &self.topology.routing_tables[node_id].neighbors;

                for p in curr_batch.drain(..) {
                    if p.header.halted {
                        halted_particles.push(p);
                    } else {
                        let mut next_hop = self.topology.routing_tables[node_id]
                            .select_next_hop(&p, self.config.temperature);

                        // P2P Decentralized Backpressure: if candidate target queue is full, overflow to neighbor in local P2P mesh
                        if self.next_queues[next_hop].len() > 64 {
                            if !neighbors.is_empty() {
                                next_hop = neighbors[p.header.hop_count as usize % neighbors.len()];
                            } else {
                                next_hop = (next_hop + 1) % self.num_nodes;
                            }
                        }
                        self.next_queues[next_hop].push(p);
                    }
                }
            }

            std::mem::swap(&mut self.node_queues, &mut self.next_queues);
        }

        // Collect any remaining in-flight particles after routing loop completes
        for q in self.node_queues.iter_mut() {
            for p in q.drain(..) {
                halted_particles.push(p);
            }
        }

        // Serializer reconstruction
        self.serializer
            .reconstruct_sequence(seq_len, &halted_particles, &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use annp_core::NormStrategy;

    fn create_test_config() -> MicroBlockConfig {
        MicroBlockConfig {
            num_shards: 4,
            mesh_rows: 2,
            mesh_cols: 2,
            d_head: 64,
            ffn_expansion: 8,
            initial_energy: 1.0,
            max_hop: 10,
            min_hop: 2,
            epsilon_p: 1e-4,
            epsilon_h: 0.05,
            temperature: 1.0,
            norm_strategy: NormStrategy::MicroRMSNorm,
            alpha_init: 0.01,
            sphere_radius: 1.0,
            lambda_temporal: 0.001,
            lambda_frequency: 0.01,
            eviction_threshold: 1e-4,
            neurogenesis_threshold: 50,
            subnode_max: 8,
            progressive_hardening_factor: 0.5,
            queue_backpressure_alpha: 0.05,
            min_routing_entropy_noise: 0.05,
            max_alpha_residual: 0.1,
        }
    }

    #[test]
    fn test_annp_model_forward_pass() -> Result<()> {
        let config = create_test_config();
        let device = Device::Cpu;
        let mut model = ANNPModel::new_with_cuda(4, 4, config, device.clone(), false);

        let d_model = 4 * 64;
        let input_tensor = Tensor::from_vec(vec![0.1f32; 2 * d_model], (2, d_model), &device)?;

        let output_tensor = model.forward(&input_tensor)?;
        assert_eq!(output_tensor.dims2()?, (2, d_model));

        Ok(())
    }
}
