//! gen_model  –  generate binary model files for seiswg examples.
//!
//! Replaces the per-example generate_model.py scripts.
//!
//! Usage:
//!   gen_model <example_dir>
//!
//! Dispatches on the directory name (forward | spin | adjoint).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::{env, fs};

// ── Binary I/O ────────────────────────────────────────────────────────────────

/// Write a seispie model binary: little-endian int32 npt header + npt × f32 payload.
fn write_bin(dir: &Path, name: &str, data: &[f32]) -> anyhow::Result<()> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("proc000000_{name}.bin"));
    let mut f = fs::File::create(&path)?;
    f.write_all(&(data.len() as i32).to_le_bytes())?;
    for &v in data {
        f.write_all(&v.to_le_bytes())?;
    }
    println!("  wrote {}", path.display());
    Ok(())
}

// ── Example generators ────────────────────────────────────────────────────────

/// Forward example: homogeneous SH model, 200×200 nodes, dx=dz=100 m.
fn gen_forward(example_dir: &Path) -> anyhow::Result<()> {
    println!("Generating forward model…");
    const NX: usize = 200;
    const NZ: usize = 200;
    const DX: f32 = 100.0;
    const DZ: f32 = 100.0;
    const NPT: usize = NX * NZ;

    let mut x = vec![0.0f32; NPT];
    let mut z = vec![0.0f32; NPT];
    for i in 0..NX {
        for j in 0..NZ {
            x[i * NZ + j] = i as f32 * DX;
            z[i * NZ + j] = j as f32 * DZ;
        }
    }

    let out = example_dir.join("model");
    write_bin(&out, "x",   &x)?;
    write_bin(&out, "z",   &z)?;
    write_bin(&out, "vp",  &vec![3000.0f32; NPT])?;
    write_bin(&out, "vs",  &vec![1732.0f32; NPT])?;
    write_bin(&out, "rho", &vec![2700.0f32; NPT])?;

    println!("\nModel: {NX}×{NZ} grid, dx={DX} m, dz={DZ} m");
    println!("npt = {NPT}");
    Ok(())
}

/// Spin example: homogeneous micropolar (Cosserat) medium, 200×200 nodes, dx=dz=1 m.
fn gen_spin(example_dir: &Path) -> anyhow::Result<()> {
    println!("Generating spin model…");
    const NX: usize = 200;
    const NZ: usize = 200;
    const DX: f32 = 1.0;
    const DZ: f32 = 1.0;
    const NPT: usize = NX * NZ;

    let mut x = vec![0.0f32; NPT];
    let mut z = vec![0.0f32; NPT];
    for i in 0..NX {
        for j in 0..NZ {
            x[i * NZ + j] = i as f32 * DX;
            z[i * NZ + j] = j as f32 * DZ;
        }
    }

    let out = example_dir.join("model");
    write_bin(&out, "x",        &x)?;
    write_bin(&out, "z",        &z)?;
    write_bin(&out, "rho",      &vec![2700.0f32; NPT])?;
    write_bin(&out, "lambda",   &vec![8.10e9f32; NPT])?;
    write_bin(&out, "mu",       &vec![8.10e9f32; NPT])?;
    write_bin(&out, "nu",       &vec![1.005e9f32; NPT])?;
    write_bin(&out, "j",        &vec![2700.0f32; NPT])?;
    write_bin(&out, "lambda_c", &vec![7.75e8f32; NPT])?;
    write_bin(&out, "mu_c",     &vec![1.50e8f32; NPT])?;
    write_bin(&out, "nu_c",     &vec![3.00e8f32; NPT])?;

    println!("\nModel: {NX}×{NZ} grid, dx={DX} m, dz={DZ} m");
    println!("npt = {NPT}");
    let vs_classical = (8.10e9f32 / 2700.0f32).sqrt();
    println!("Classical S-wave speed ≈ {vs_classical:.1} m/s");
    Ok(())
}

/// Adjoint example: homogeneous model_init + checkerboard model_true, 200×200 nodes, dx=dz=2400 m.
fn gen_adjoint(example_dir: &Path) -> anyhow::Result<()> {
    const NX: usize = 200;
    const NZ: usize = 200;
    const DX: f32 = 2400.0;
    const DZ: f32 = 2400.0;
    const NPT: usize = NX * NZ;

    const VP0: f32 = 5500.0;
    const VS0: f32 = 3500.0;
    const RHO0: f32 = 2600.0;

    // Checkerboard: cell = 40 grid pts (96 km), ±11.4% vs perturbation.
    const CELL: f32 = 40.0;
    const DVS_FRAC: f32 = 0.114;
    let kx = std::f32::consts::PI / CELL;
    let kz = std::f32::consts::PI / CELL;

    let mut x        = vec![0.0f32; NPT];
    let mut z        = vec![0.0f32; NPT];
    let mut vs_true  = vec![0.0f32; NPT];

    for i in 0..NX {
        for j in 0..NZ {
            let k = i * NZ + j;
            x[k] = i as f32 * DX;
            z[k] = j as f32 * DZ;
            let checker = (kx * i as f32).sin() * (kz * j as f32).sin();
            vs_true[k] = VS0 * (1.0 + DVS_FRAC * checker);
        }
    }

    // model_init – homogeneous
    println!("Writing model_init/ …");
    let init_dir = example_dir.join("model_init");
    write_bin(&init_dir, "vp",  &vec![VP0;  NPT])?;
    write_bin(&init_dir, "vs",  &vec![VS0;  NPT])?;
    write_bin(&init_dir, "rho", &vec![RHO0; NPT])?;
    write_bin(&init_dir, "x",   &x)?;
    write_bin(&init_dir, "z",   &z)?;

    // model_true – checkerboard vs perturbation (plus coordinates for plotting)
    println!("Writing model_true/ …");
    let true_dir = example_dir.join("model_true");
    write_bin(&true_dir, "vp",  &vec![VP0;  NPT])?;
    write_bin(&true_dir, "vs",  &vs_true)?;
    write_bin(&true_dir, "rho", &vec![RHO0; NPT])?;
    write_bin(&true_dir, "x",   &x)?;
    write_bin(&true_dir, "z",   &z)?;

    let domain_km = NX as f32 * DX / 1000.0;
    println!("\nModel: {NX}×{NZ} grid, dx={DX} m, dz={DZ} m  ({domain_km} km × {domain_km} km)");
    println!("npt = {NPT}");
    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: gen_model <example_dir>");
        std::process::exit(1);
    }

    let example_dir = PathBuf::from(&args[1]);
    let name = example_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    match name {
        "forward" => gen_forward(&example_dir)?,
        "spin"    => gen_spin(&example_dir)?,
        "adjoint" => gen_adjoint(&example_dir)?,
        other => {
            eprintln!("Unknown example '{other}'. Expected: forward | spin | adjoint");
            std::process::exit(1);
        }
    }

    Ok(())
}
