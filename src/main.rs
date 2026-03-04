//! seispie-wg  –  2-D finite-difference seismic solver, Rust + WebGPU.
//!
//! Usage:
//!   seispie-wg <config.ini>
//!
//! The config.ini format is identical to the Python seispie configs.

mod config;
mod solver;

use std::path::PathBuf;
use anyhow::{Context, Result};

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.ini>", args[0]);
        std::process::exit(1);
    }
    let config_path = PathBuf::from(&args[1]);

    log::info!("Loading config from {}", config_path.display());
    let cfg = config::load(&config_path)
        .with_context(|| format!("Failed to load {}", config_path.display()))?;

    log::info!("Workflow: {}", cfg.workflow);

    let mut slv = solver::Solver::new(cfg.clone())
        .context("Failed to initialise solver")?;

    match cfg.workflow.as_str() {
        "forward" => {
            slv.run_forward()?;
        }
        other => {
            anyhow::bail!("Workflow '{}' is not yet implemented in seispie-wg", other);
        }
    }

    log::info!("Done.");
    Ok(())
}
