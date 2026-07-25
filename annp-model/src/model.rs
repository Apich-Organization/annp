use crate::micro_block::MicroBlockNode;
use crate::scattering::TokenScattering;
use crate::serializer::EgressSerializer;
use crate::topology::TopologyGrid;
use annp_core::{MicroBlockConfig, Particle};
use candle_core::{Device, Result, Tensor};

/// Full ANNP Architecture Pipeline.
pub struct ANNPModel {
    pub config: MicroBlockConfig,
    pub num_nodes: usize,
    pub scattering: TokenScattering,
    pub nodes: Vec<MicroBlockNode>,
    pub topology: TopologyGrid,
    pub serializer: EgressSerializer,
    pub device: Device,
}

impl ANNPModel {
    pub fn new(
        num_nodes: usize,
        num_shards: usize,
        config: MicroBlockConfig,
        device: Device,
    ) -> Self {
        let d_head = config.d_head;
        let scattering = TokenScattering::new(num_shards, d_head, 0.1);
        let topology = TopologyGrid::new(num_nodes, d_head, 4);
        let serializer = EgressSerializer::new(d_head, num_shards);

        let nodes = (0..num_nodes)
            .map(|i| MicroBlockNode::new(i, config.clone(), 64))
            .collect();

        Self {
            config,
            num_nodes,
            scattering,
            nodes,
            topology,
            serializer,
            device,
        }
    }

    /// Forward pass through ANNP P2P Mesh with lock-free asynchronous particle routing
    pub fn forward(&mut self, embeddings: &Tensor) -> Result<Tensor> {
        let (seq_len, _) = embeddings.dims2()?;
        let initial_particles = self
            .scattering
            .scatter_embeddings(embeddings, &self.config)?;

        let mut node_queues: Vec<Vec<Particle>> = vec![Vec::new(); self.num_nodes];

        // Ingress distribution
        let ingress_nodes = &self.scattering.ingress_node_indices;
        for (idx, p) in initial_particles.into_iter().enumerate() {
            let target_ingress = ingress_nodes[idx % ingress_nodes.len()];
            node_queues[target_ingress].push(p);
        }

        let mut halted_particles: Vec<Particle> = Vec::new();
        let mut active_loop = true;
        let mut step = 0;
        let max_steps = self.config.max_hop as usize + 20;

        while active_loop && step < max_steps {
            step += 1;
            active_loop = false;
            let mut next_queues: Vec<Vec<Particle>> = vec![Vec::new(); self.num_nodes];

            for node_id in 0..self.num_nodes {
                let mut curr_batch = std::mem::take(&mut node_queues[node_id]);
                if curr_batch.is_empty() {
                    continue;
                }
                active_loop = true;

                // Process Micro-Block computing
                self.nodes[node_id].process_batch(&mut curr_batch);

                // Push-routing decision for output particles
                for p in curr_batch {
                    if p.header.halted {
                        halted_particles.push(p);
                    } else {
                        let next_hop = self.topology.routing_tables[node_id]
                            .select_next_hop(&p, self.config.temperature);
                        next_queues[next_hop].push(p);
                    }
                }
            }

            node_queues = next_queues;
        }

        // Serializer reconstruction
        self.serializer
            .reconstruct_sequence(seq_len, &halted_particles, &self.device)
    }
}
