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
    /// Train ANNP model through 4-stage evolutionary training with loss convergence tracking
    Train {
        /// Configuration TOML path
        #[arg(short, long, default_value = "annp_config.toml")]
        config: PathBuf,
        /// Target training stage: "0", "1", "2", "3", or "all"
        #[arg(short, long, default_value = "all")]
        stage: String,
        /// Optional path to resume training from a checkpoint file
        #[arg(short, long)]
        resume_from: Option<PathBuf>,
        /// Directory to save output model checkpoints
        #[arg(short, long, default_value = "checkpoints")]
        output_dir: PathBuf,
    },
    /// Run model inference pass and throughput performance benchmarks
    Run {
        /// Configuration TOML path
        #[arg(short, long, default_value = "annp_config.toml")]
        config: PathBuf,
        /// Optional checkpoint path to load trained model weights
        #[arg(short = 'k', long)]
        checkpoint: Option<PathBuf>,
        /// Optional input string / token sequence
        #[arg(short, long)]
        input: Option<String>,
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
            output_dir,
        } => execute_train(config, stage, resume_from, output_dir)?,
        Commands::Run {
            config,
            checkpoint,
            input,
            benchmark,
        } => execute_run(config, checkpoint, input, benchmark)?,
        Commands::Export { checkpoint, out } => execute_export(checkpoint, out)?,
    }

    Ok(())
}
