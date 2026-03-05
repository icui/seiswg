//! seiswg  –  2-D finite-difference seismic solver, Rust + WebGPU.
//!
//! Usage:
//!   seiswg <config.ini>
//!
//! The config.ini format is identical to the Python seispie configs.

mod config;
mod solver;

use std::collections::HashMap;
use std::path::PathBuf;
use anyhow::{Context, Result};
use solver::Vfs;

/// Recursively load all files under `dir` into a VFS, keyed by their path string.
fn load_dir_into_vfs(vfs: &mut Vfs, dir: &std::path::Path) -> Result<()> {
    if !dir.exists() { return Ok(()); }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            load_dir_into_vfs(vfs, &path)?;
        } else {
            let bytes = std::fs::read(&path)
                .with_context(|| format!("Cannot read {}", path.display()))?;
            vfs.insert(path.display().to_string(), bytes);
        }
    }
    Ok(())
}

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

    // Build a VFS from the model directory and sources/stations files.
    let mut vfs: Vfs = HashMap::new();
    if let Some(model_dir) = cfg.paths.model_init.as_deref()
        .or(cfg.paths.model_true.as_deref())
    {
        load_dir_into_vfs(&mut vfs, model_dir)
            .with_context(|| format!("Loading model files from {}", model_dir.display()))?;
    }
    // Also load a model_true if running adjoint
    if let Some(mt) = cfg.paths.model_true.as_deref() {
        load_dir_into_vfs(&mut vfs, mt).ok();
    }
    // Load sources and stations
    if cfg.paths.sources.exists() {
        vfs.insert(
            cfg.paths.sources.display().to_string(),
            std::fs::read(&cfg.paths.sources)?,
        );
    }
    if cfg.paths.stations.exists() {
        vfs.insert(
            cfg.paths.stations.display().to_string(),
            std::fs::read(&cfg.paths.stations)?,
        );
    }
    // For adjoint, pre-load observed traces directory
    if let Some(obs_dir) = cfg.paths.traces.as_deref() {
        load_dir_into_vfs(&mut vfs, obs_dir).ok();
    }

    pollster::block_on(async {
        let mut slv = solver::Solver::new(cfg.clone(), &vfs)
            .await
            .context("Failed to initialise solver")?;

        match cfg.workflow.as_str() {
            "forward" => {
                slv.run_forward().await?;
            }
            "adjoint" => {
                slv.run_adjoint(&vfs).await?;
            }
            other => {
                anyhow::bail!("Workflow '{}' is not yet implemented in seiswg", other);
            }
        }
        log::info!("Done.");
        Ok::<(), anyhow::Error>(())
    })
}
