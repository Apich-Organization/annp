mod checkpoint;
mod commands;
mod config;
mod dataset;

use clap::{Parser, Subcommand};
use commands::{execute_export, execute_init, execute_run, execute_train};
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
    /// Train ANNP model through 2-stage evolutionary training with loss convergence tracking
    Train {
        /// Configuration TOML path
        #[arg(short, long, default_value = "annp_config.toml")]
        config: PathBuf,
        /// Target training stage: "0" (exploration), "1" (hardening), or "all"
        #[arg(short, long, default_value = "all")]
        stage: String,
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
    },
    /// Export P2P mesh topology and Q-Routing tables from checkpoint
    Export {
        /// Checkpoint file path
        #[arg(short, long)]
        checkpoint: PathBuf,
        /// Output JSON path for topology routing tables
        #[arg(short, long, default_value = "topology_routing.json")]
        out: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Init { output } => execute_init(output)?,
        Commands::Train {
            config,
            stage,
            resume_from,
            format,
            device,
            output_dir,
        } => execute_train(config, stage, resume_from, format, device, output_dir)?,
        Commands::Run {
            config,
            checkpoint,
            input,
            temperature,
            device,
            continual,
            save_output,
            benchmark,
        } => execute_run(
            config,
            checkpoint,
            input,
            temperature,
            device,
            continual,
            save_output,
            benchmark,
        )?,
        Commands::Export { checkpoint, out } => execute_export(checkpoint, out)?,
    }

    Ok(())
}
