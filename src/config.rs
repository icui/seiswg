use anyhow::{anyhow, Context, Result};
use configparser::ini::Ini;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GridConfig {
    pub nt: u32,
    pub dt: f32,
}

#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    pub left:   bool,
    pub right:  bool,
    pub bottom: bool,
    pub top:    bool,
    pub width:  u32,
    pub alpha:  f32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SolverConfig {
    pub sh:   bool,
    pub psv:  bool,
    pub spin: bool,
    pub threads_per_block: u32,
    pub combine_sources:   bool,
    pub save_snapshot:     u32,
    pub adj_interval:      u32,
    pub smooth:            f32,
    pub grid:     GridConfig,
    pub boundary: BoundaryConfig,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PathConfig {
    pub output:         PathBuf,
    pub output_traces:  Option<PathBuf>,
    pub model_true:     Option<PathBuf>,
    pub model_init:     Option<PathBuf>,
    pub sources:        PathBuf,
    pub stations:       PathBuf,
    pub traces:         Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub workflow: String,   // "forward" | "adjoint" | "inversion"
    pub solver:   SolverConfig,
    pub paths:    PathConfig,
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn get_str(ini: &Ini, section: &str, key: &str) -> Option<String> {
    ini.get(section, key)
}

fn get_required(ini: &Ini, section: &str, key: &str) -> Result<String> {
    ini.get(section, key)
        .ok_or_else(|| anyhow!("[{}] {} is required", section, key))
}

fn get_bool(ini: &Ini, section: &str, key: &str, default: bool) -> bool {
    ini.get(section, key)
        .map(|v| v.trim().to_lowercase() == "yes")
        .unwrap_or(default)
}

fn get_u32(ini: &Ini, section: &str, key: &str) -> Result<u32> {
    get_required(ini, section, key)?
        .trim()
        .parse::<u32>()
        .with_context(|| format!("[{}] {} must be an integer", section, key))
}

fn get_u32_or(ini: &Ini, section: &str, key: &str, default: u32) -> u32 {
    ini.get(section, key)
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(default)
}

fn get_f32(ini: &Ini, section: &str, key: &str) -> Result<f32> {
    get_required(ini, section, key)?
        .trim()
        .parse::<f32>()
        .with_context(|| format!("[{}] {} must be a float", section, key))
}

fn get_f32_or(ini: &Ini, section: &str, key: &str, default: f32) -> f32 {
    ini.get(section, key)
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(default)
}

// Resolve a path relative to the directory containing the config file.
fn resolve(base: &Path, raw: &str) -> PathBuf {
    let p = Path::new(raw.trim());
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn load(config_path: &Path) -> Result<Config> {
    let mut ini = Ini::new();
    ini.load(config_path)
        .map_err(|e| anyhow!("Cannot open {}: {}", config_path.display(), e))?;

    let base = config_path
        .parent()
        .unwrap_or_else(|| Path::new("."));

    // ── workflow ────────────────────────────────────────────────────────────
    let workflow = get_required(&ini, "workflow", "mode")?
        .trim()
        .to_lowercase();

    // ── solver ──────────────────────────────────────────────────────────────
    let grid = GridConfig {
        nt: get_u32(&ini, "solver", "nt")?,
        dt: get_f32(&ini, "solver", "dt")?,
    };

    let boundary = BoundaryConfig {
        left:   get_bool(&ini, "solver", "abs_left",   false),
        right:  get_bool(&ini, "solver", "abs_right",  false),
        bottom: get_bool(&ini, "solver", "abs_bottom", false),
        top:    get_bool(&ini, "solver", "abs_top",    false),
        width:  get_u32_or(&ini, "solver", "abs_width", 20),
        alpha:  get_f32_or(&ini, "solver", "abs_alpha", 0.015),
    };

    let solver = SolverConfig {
        sh:               get_bool(&ini, "solver", "sh",  false),
        psv:              get_bool(&ini, "solver", "psv", true),
        spin:             get_bool(&ini, "solver", "spin", false),
        threads_per_block: get_u32_or(&ini, "solver", "threads_per_block", 128),
        combine_sources:  get_bool(&ini, "solver", "combine_sources", false),
        save_snapshot:    get_u32_or(&ini, "solver", "save_snapshot", 0),
        adj_interval:     get_u32_or(&ini, "solver", "adj_interval", 10),
        smooth:           get_f32_or(&ini, "solver", "smooth", 5.0),
        grid,
        boundary,
    };

    // ── paths ───────────────────────────────────────────────────────────────
    let paths = PathConfig {
        output: resolve(base,
            &get_required(&ini, "path", "output")?),
        output_traces: get_str(&ini, "path", "output_traces")
            .map(|s| resolve(base, &s)),
        model_true: get_str(&ini, "path", "model_true")
            .map(|s| resolve(base, &s)),
        model_init: get_str(&ini, "path", "model_init")
            .map(|s| resolve(base, &s)),
        sources:  resolve(base, &get_required(&ini, "path", "sources")?),
        stations: resolve(base, &get_required(&ini, "path", "stations")?),
        traces:   get_str(&ini, "path", "traces")
            .map(|s| resolve(base, &s)),
    };

    Ok(Config { workflow, solver, paths })
}
