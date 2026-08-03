#![allow(clippy::too_many_arguments)]

mod checkpoint;
mod commands;
mod config;
mod dataset;
pub mod logger;
pub mod tokenizer;

use clap::{Parser, Subcommand};
use commands::{execute_edit_model, execute_export, execute_init, execute_run, execute_train};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "annp")]
#[command(about = "Industrial-Standard Asynchronous Neural Network Protocol (ANNP) CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize default ANNP TOML configuration file
    Init {
        /// Output path for configuration TOML
        #[arg(short, long, default_value = "annp_config.toml")]
        output: PathBuf,
    },
    /// Train ANNP model with loss convergence tracking
    Train {
        /// Configuration TOML path
        #[arg(short, long, default_value = "annp_config.toml")]
        config: PathBuf,
        /// Optional path to resume training from a checkpoint file (.annpb binary or .json)
        #[arg(short, long)]
        resume_from: Option<PathBuf>,
        /// Checkpoint format: "annpb" (high-performance binary) or "json"
        #[arg(short = 'f', long, default_value = "annpb")]
        format: String,
        /// Target execution device: "auto", "cpu", or "cuda"
        #[arg(short = 'd', long, default_value = "auto")]
        device: String,
        /// Directory to save output model checkpoints
        #[arg(short, long, default_value = "checkpoints")]
        output_dir: PathBuf,
        /// Directory to save execution log files
        #[arg(long, default_value = "logs")]
        log_dir: PathBuf,
    },
    /// Run model inference pass and throughput performance benchmarks
    Run {
        /// Configuration TOML path
        #[arg(short, long, default_value = "annp_config.toml")]
        config: PathBuf,
        /// Optional checkpoint path (.annpb binary or .json) to load trained model weights
        #[arg(short = 'k', long)]
        checkpoint: Option<PathBuf>,
        /// Optional input vector / token sequence
        #[arg(short, long)]
        input: Option<String>,
        /// Runtime routing temperature override (\tau > 0)
        #[arg(short = 't', long)]
        temperature: Option<f32>,
        /// Target execution device: "auto", "cpu", or "cuda"
        #[arg(short = 'd', long, default_value = "auto")]
        device: String,
        /// Enable Online Continual Learning mode during inference (updates node activation counts & plastic hardening)
        #[arg(long)]
        continual: bool,
        /// Save output sequence tensor to binary file (.annpb)
        #[arg(short, long)]
        save_output: Option<PathBuf>,
        /// Enable high-throughput particle processing benchmark
        #[arg(short, long)]
        benchmark: bool,
        /// Directory to save execution log files
        #[arg(long, default_value = "logs")]
        log_dir: PathBuf,
    },
    /// Edit model checkpoint configuration headers with automatic backup (.bak)
    EditModel {
        /// Path to model checkpoint file (.annpb or .json)
        #[arg(short = 'k', long)]
        checkpoint: PathBuf,
        /// Optional path to new TOML configuration file to apply
        #[arg(short = 'c', long)]
        config: Option<PathBuf>,
        /// Override max_hop ceiling
        #[arg(long)]
        max_hop: Option<u16>,
        /// Override min_hop floor
        #[arg(long)]
        min_hop: Option<u16>,
        /// Override initial particle energy
        #[arg(long)]
        initial_energy: Option<f32>,
        /// Override weight decay factor
        #[arg(long)]
        weight_decay: Option<f32>,
        /// Override negative credit damping factor for subnodes
        #[arg(long)]
        negative_credit_damping: Option<f32>,
        /// Override early halting consecutive negative credit streak threshold
        #[arg(long)]
        early_halt_streak: Option<usize>,
        /// Override positional encoding scale factor
        #[arg(long)]
        pos_enc_scale: Option<f32>,
        /// Override positional encoding base frequency
        #[arg(long)]
        pos_base_freq: Option<f32>,
        /// Override completed epoch count
        #[arg(short = 'e', long)]
        epoch: Option<usize>,
        /// Override completed stage count
        #[arg(short = 's', long)]
        stage: Option<usize>,
        /// Reset transient TD particle state (last_p_in, last_prediction, last_token_id)
        #[arg(long)]
        reset_state: bool,
        /// Reset runtime statistics (activation counts, credit stats, node health)
        #[arg(long)]
        reset_stats: bool,
        /// Reset Hebbian associative memory matrices (fast_weight) and cumulative energy
        #[arg(long)]
        reset_fast_weights: bool,
        /// Reset routing table weights and edge credit statistics
        #[arg(long)]
        reset_routing: bool,
    },
    /// Export P2P mesh topology and Q-Routing tables from checkpoint (JSON, DOT, CSV, Summary)
    Export {
        /// Checkpoint file path (.annpb or .json)
        #[arg(short = 'k', long)]
        checkpoint: PathBuf,
        /// Output file path
        #[arg(short = 'o', long, default_value = "topology_routing.json")]
        out: PathBuf,
        /// Export format: "json", "dot" (Graphviz), "csv" (Edge list), or "summary" (auto-detected if omitted)
        #[arg(short = 'f', long)]
        format: Option<String>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { output } => execute_init(output)?,
        Commands::Train {
            config,
            resume_from,
            format,
            device,
            output_dir,
            log_dir,
        } => execute_train(config, resume_from, format, device, output_dir, log_dir)?,
        Commands::Run {
            config,
            checkpoint,
            input,
            temperature,
            device,
            continual,
            save_output,
            benchmark,
            log_dir,
        } => execute_run(
            config,
            checkpoint,
            input,
            temperature,
            device,
            continual,
            save_output,
            benchmark,
            log_dir,
        )?,
        Commands::EditModel {
            checkpoint,
            config,
            max_hop,
            min_hop,
            initial_energy,
            weight_decay,
            negative_credit_damping,
            early_halt_streak,
            pos_enc_scale,
            pos_base_freq,
            epoch,
            stage,
            reset_state,
            reset_stats,
            reset_fast_weights,
            reset_routing,
        } => execute_edit_model(
            checkpoint,
            config,
            max_hop,
            min_hop,
            initial_energy,
            weight_decay,
            negative_credit_damping,
            early_halt_streak,
            pos_enc_scale,
            pos_base_freq,
            epoch,
            stage,
            reset_state,
            reset_stats,
            reset_fast_weights,
            reset_routing,
        )?,
        Commands::Export {
            checkpoint,
            out,
            format,
        } => execute_export(checkpoint, out, format)?,
    }

    Ok(())
}
