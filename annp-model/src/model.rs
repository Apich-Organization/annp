use crate::micro_block::MicroBlockNode;
use crate::scattering::TokenScattering;
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
    pub device: Device,
    pub is_training: bool,
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
            device,
            is_training: true,
            node_queues,
            next_queues,
        }
    }

    pub fn reset_state(&mut self) {
        for node in self.nodes.iter_mut() {
            node.k_cache.clear();
            node.v_cache.clear();
            node.kv_traces.clear();
            node.kv_token_ids.clear();
            node.kv_shard_ids.clear();
            node.last_p_in.iter_mut().for_each(|x| *x = 0.0);
            node.recent_activation_count = 0;
            node.local_loss_accumulator = 0.0;
            node.local_loss_count = 0;
        }
    }

    /// Forward pass through ANNP P2P Mesh with lock-free dtact coroutine mesh scheduling
    pub fn forward(&mut self, embeddings: &Tensor, offset: usize) -> Result<(Tensor, f32)> {
        let (seq_len, _) = embeddings.dims2()?;
        let mut initial_particles =
            self.scattering
                .scatter_embeddings(embeddings, &self.config, offset)?;
        for particle in &mut initial_particles {
            particle.trace_concentration = 1.0;
        }

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

            let use_cuda_mode = self.nodes.first().is_some_and(|n| n.use_cuda);

            if use_cuda_mode {
                // GPU Mode: Execute active nodes on CUDA GPU stream without CPU multi-threading contention
                for (node, batch) in self.nodes.iter_mut().zip(curr_batches.iter_mut()) {
                    if !batch.is_empty() {
                        node.process_batch(batch, self.is_training);
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

                        let is_train = self.is_training;
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
                                (*node_ptr).process_batch(&mut *b_ptr, is_train);
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

                // Clone this small local adjacency list so a route can be
                // reinforced below without holding an immutable table borrow.
                let neighbors = self.topology.routing_tables[node_id].neighbors.clone();

                for p in curr_batch.drain(..) {
                    if p.header.halted {
                        halted_particles.push(p);
                    } else {
                        let mut next_hop =
                            self.topology.routing_tables[node_id].select_next_hop(&p);

                        // P2P Decentralized Backpressure: if candidate target queue is full, overflow to neighbor in local P2P mesh
                        if self.next_queues[next_hop].len() > 64 {
                            if !neighbors.is_empty() {
                                next_hop = neighbors[p.header.hop_count as usize % neighbors.len()];
                            } else {
                                next_hop = (next_hop + 1) % self.num_nodes;
                            }
                        }
                        if p.credit_valid {
                            self.topology.routing_tables[node_id]
                                .observe_credit(next_hop, p.credit);
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

        // Fully Decentralized Reconstruction:
        // Emulate motor nodes emitting their state without a centralized dense projector.
        let d_model = self.config.d_head * self.config.num_shards;
        let mut full_data = vec![0.0f32; seq_len * d_model];

        for p in halted_particles {
            let t = p.header.origin_token_id as usize;
            let shard = p.header.shard_id as usize;

            if t < seq_len && shard < self.config.num_shards {
                let token_offset = t * d_model;
                let shard_offset = token_offset + shard * self.config.d_head;

                for d in 0..self.config.d_head {
                    if d < p.payload.len() {
                        full_data[shard_offset + d] = p.payload[d];
                    }
                }
            }
        }

        // Global Micro-RMS bounding was removed.
        // We now rely on intra-node Pre-RMSNorm and exponential moving average attention
        // to maintain stability, allowing the network to express proportional amplitude.

        let out_tensor = Tensor::from_vec(full_data, (seq_len, d_model), &self.device)?;

        let mut total_loss = 0.0;
        let mut total_count = 0;
        for node in &self.nodes {
            total_loss += node.local_loss_accumulator;
            total_count += node.local_loss_count;
        }
        let avg_loss = if total_count > 0 {
            total_loss / total_count as f32
        } else {
            0.0
        };

        Ok((out_tensor, avg_loss))
    }

    pub fn extract_batch_metrics(&mut self) -> BatchMetrics {
        let mut total_hops = 0;
        let mut total_halted = 0;
        let mut total_energy = 0.0;
        let mut total_particles = 0;
        let mut total_entropy = 0.0;
        let mut total_attn_ops = 0;
        let mut total_volatility = 0.0;
        let mut total_affinity = 0.0;
        let mut total_subnodes = 0;

        let mut utilization = Vec::with_capacity(self.num_nodes);

        for node in self.nodes.iter_mut() {
            let nm = node.extract_and_reset_metrics();
            total_hops += nm.sum_hop_count;
            total_halted += nm.halted_particles_count;
            total_energy += nm.sum_squared_energy;
            total_particles += nm.total_particles_processed;
            total_entropy += nm.sum_attention_entropy;
            total_attn_ops += nm.attention_ops_count;
            total_volatility += nm.sum_credit_volatility;
            total_affinity += nm.sum_temporal_affinity;
            total_subnodes += nm.active_subnodes_count;

            utilization.push(nm.total_particles_processed);
        }

        utilization.sort_unstable();
        let n = utilization.len() as f32;
        let mut num = 0.0;
        let mut den = 0.0;
        for (i, &y) in utilization.iter().enumerate() {
            let y_f = y as f32;
            num += (i as f32 + 1.0) * y_f;
            den += y_f;
        }

        let gini = if den > 0.0 && n > 0.0 {
            (2.0 * num) / (n * den) - (n + 1.0) / n
        } else {
            0.0
        };

        let pt = total_particles.max(1) as f32;
        BatchMetrics {
            avg_hop_count: total_hops as f32 / pt,
            early_halting_rate: total_halted as f32 / pt,
            avg_signal_energy: total_energy / pt,
            avg_subnodes: total_subnodes as f32 / self.num_nodes.max(1) as f32,
            utilization_gini: gini,
            avg_attention_entropy: if total_attn_ops > 0 {
                total_entropy / total_attn_ops as f32
            } else {
                0.0
            },
            avg_credit_volatility: total_volatility / pt,
            avg_temporal_affinity: total_affinity / pt,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct BatchMetrics {
    pub avg_hop_count: f32,
    pub early_halting_rate: f32,
    pub avg_signal_energy: f32,
    pub avg_subnodes: f32,
    pub utilization_gini: f32,
    pub avg_attention_entropy: f32,
    pub avg_credit_volatility: f32,
    pub avg_temporal_affinity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

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
            subnode_max: 8,
        }
    }

    #[test]
    fn test_annp_model_forward_pass() -> Result<()> {
        let config = create_test_config();
        let device = Device::Cpu;
        let mut model = ANNPModel::new_with_cuda(4, 4, config, device.clone(), false);

        let d_model = 4 * 64;
        let input_tensor = Tensor::from_vec(vec![0.1f32; 2 * d_model], (2, d_model), &device)?;

        let (output_tensor, loss) = model.forward(&input_tensor, 0)?;
        assert_eq!(output_tensor.dims2()?, (2, d_model));
        assert!(loss >= 0.0);

        Ok(())
    }
}
