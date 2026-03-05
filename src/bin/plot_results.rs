//! plot_results  –  visualise seiswg output, PNG output.
//!
//! Replaces scripts/plot_results.py.
//!
//! Usage:
//!   plot_results <example_dir>

use std::io::Read;
use std::path::{Path, PathBuf};
use std::{env, fs};

use plotters::prelude::*;

// ── Binary I/O ────────────────────────────────────────────────────────────────

/// Read a seispie model binary: little-endian int32 npt header + npt×f32 payload.
fn read_bin(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut buf = Vec::new();
    fs::File::open(path)?.read_to_end(&mut buf)?;
    let npt = i32::from_le_bytes(buf[..4].try_into()?) as usize;
    let data: Vec<f32> = buf[4..4 + npt * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(data)
}

/// Read a raw float32 snapshot file (no header) written by solver.rs.
fn read_raw_bin(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut buf = Vec::new();
    fs::File::open(path)?.read_to_end(&mut buf)?;
    let data: Vec<f32> = buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(data)
}

// ── NPY reader (2-D float32 arrays only) ─────────────────────────────────────

/// Minimal .npy reader.  Returns `(shape, flat_data)` for 2-D f32 arrays.
/// Returns `None` for 1-D or non-f32 arrays (silently skipped by callers).
fn read_npy(path: &Path) -> anyhow::Result<Option<(usize, usize, Vec<f32>)>> {
    let mut buf = Vec::new();
    fs::File::open(path)?.read_to_end(&mut buf)?;

    // Magic: 0x93 N U M P Y
    if buf.len() < 10 || buf[0] != 0x93 || &buf[1..6] != b"NUMPY" {
        anyhow::bail!("Not a numpy file: {}", path.display());
    }
    let major = buf[6];
    let (hdr_len, data_offset) = if major == 1 {
        (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10usize)
    } else {
        (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12usize)
    };
    let header = std::str::from_utf8(&buf[data_offset..data_offset + hdr_len])?;

    // Only handle little-endian float32
    if !header.contains("'<f4'") && !header.contains("\"<f4\"") {
        return Ok(None);
    }
    // Only handle C-order (not Fortran order)
    if header.contains("'fortran_order': True") {
        return Ok(None);
    }

    // Parse shape
    let shape = parse_npy_shape(header)?;
    if shape.len() != 2 {
        return Ok(None); // skip 1-D or >2-D
    }
    let (nrec, nt) = (shape[0], shape[1]);

    let start = data_offset + hdr_len;
    let nelem = nrec * nt;
    let data: Vec<f32> = buf[start..start + nelem * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    Ok(Some((nrec, nt, data)))
}

fn parse_npy_shape(header: &str) -> anyhow::Result<Vec<usize>> {
    let start = header
        .find("'shape'")
        .ok_or_else(|| anyhow::anyhow!("No 'shape' key in NPY header"))?;
    let after = &header[start + "'shape'".len()..];
    let po = after
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("No opening paren after 'shape'"))?;
    let inner = &after[po + 1..];
    let pc = inner
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("No closing paren for shape"))?;
    let shape: Vec<usize> = inner[..pc]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    Ok(shape)
}

// ── Colormaps ─────────────────────────────────────────────────────────────────

fn lerp3(a: [u8; 3], b: [u8; 3], t: f32) -> RGBColor {
    RGBColor(
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t).clamp(0.0, 255.0) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t).clamp(0.0, 255.0) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t).clamp(0.0, 255.0) as u8,
    )
}

fn interp_palette(keys: &[[u8; 3]], t: f32) -> RGBColor {
    let n = keys.len() - 1;
    let t = t.clamp(0.0, 1.0);
    let seg = (t * n as f32).min(n as f32 - 1e-6) as usize;
    let frac = t * n as f32 - seg as f32;
    lerp3(keys[seg], keys[seg + 1], frac)
}

/// Diverging blue–white–red colormap (matplotlib "seismic").
fn seismic(t: f32) -> RGBColor {
    const KEYS: [[u8; 3]; 5] = [
        [50, 35, 129],   // t=0.00  dark blue
        [97, 138, 218],  // t=0.25  cornflower
        [255, 255, 255], // t=0.50  white
        [220, 93, 55],   // t=0.75  tomato
        [152, 0, 0],     // t=1.00  dark red
    ];
    interp_palette(&KEYS, t)
}

/// Sequential purple→blue→green→yellow colormap (matplotlib "viridis").
fn viridis(t: f32) -> RGBColor {
    const KEYS: [[u8; 3]; 5] = [
        [68, 1, 84],
        [58, 82, 139],
        [32, 144, 141],
        [94, 200, 97],
        [253, 231, 37],
    ];
    interp_palette(&KEYS, t)
}

// ── Axis-tick label formatter ────────────────────────────────────────────────

/// Format a float for an axis tick: scientific notation when |v| >= 1e4 or
/// (non-zero and |v| < 0.01), otherwise plain decimal with up to 4 sig figs.
fn sci_label(v: f32) -> String {
    let av = v.abs();
    if av == 0.0 {
        return "0".to_string();
    }
    if av >= 1e4 || av < 0.01 {
        // e.g. 4.80e5  or  3.50e-3
        let exp = av.log10().floor() as i32;
        let mantissa = v / 10f32.powi(exp);
        format!("{mantissa:.2}e{exp}")
    } else {
        // plain, trim trailing zeros
        let s = format!("{v:.4}");
        let s = s.trim_end_matches('0');
        s.trim_end_matches('.').to_string()
    }
}

// ── Grid helpers ──────────────────────────────────────────────────────────────

fn fmin(v: &[f32]) -> f32 { v.iter().cloned().fold(f32::INFINITY, f32::min) }
fn fmax(v: &[f32]) -> f32 { v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) }

/// Given coordinate arrays (arbitrary ordering), compute (nx, nz) from the
/// bounding-box aspect ratio, matching the logic in plot_results.py.
fn infer_grid(x: &[f32], z: &[f32]) -> (usize, usize) {
    let npt = x.len().min(z.len());
    let lx = fmax(x) - fmin(x);
    let lz = fmax(z) - fmin(z);
    if lx == 0.0 || lz == 0.0 {
        return (npt, 1);
    }
    let nx = ((npt as f32 * lx / lz).sqrt()).round() as usize;
    let nz = (npt as f32 / nx.max(1) as f32).round() as usize;
    let nx = (npt / nz.max(1)).min(npt);
    let nz = npt / nx.max(1);
    (nx, nz)
}

// ── Simple colorbar drawn as a vertical band on the right ────────────────────

fn draw_colorbar<DB>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    cmap: fn(f32) -> RGBColor,
    vmin: f32,
    vmax: f32,
    label: &str,
) -> anyhow::Result<()>
where
    DB: DrawingBackend,
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let (w, h) = root.dim_in_pixel();
    // Bar sits in the right margin; leave enough room for tick labels (≈100 px)
    let bar_x   = (w as i32) - 90;
    let bar_w   = 22i32;
    let bar_y_top = 50i32;
    let bar_y_bot = (h as i32) - 70;
    let bar_h = (bar_y_bot - bar_y_top).max(1) as u32;

    for row in 0..bar_h {
        let t = 1.0 - row as f32 / bar_h as f32;
        let color = cmap(t);
        root.draw(&Rectangle::new(
            [
                (bar_x, bar_y_top + row as i32),
                (bar_x + bar_w, bar_y_top + row as i32 + 1),
            ],
            color.filled(),
        ))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }
    // Border
    root.draw(&Rectangle::new(
        [(bar_x, bar_y_top), (bar_x + bar_w, bar_y_bot)],
        BLACK.stroke_width(1),
    ))
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let font = ("sans-serif", 22).into_font();
    // Max label (top)
    root.draw(&Text::new(
        sci_label(vmax),
        (bar_x + bar_w + 4, bar_y_top - 1),
        font.clone(),
    ))
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // Mid label
    let vmid = (vmin + vmax) * 0.5;
    root.draw(&Text::new(
        sci_label(vmid),
        (bar_x + bar_w + 4, bar_y_top + (bar_h / 2) as i32 - 8),
        font.clone(),
    ))
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // Min label (bottom)
    root.draw(&Text::new(
        sci_label(vmin),
        (bar_x + bar_w + 4, bar_y_bot - 18),
        font.clone(),
    ))
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    // Optional axis label (rotated text not supported; place above bar)
    if !label.is_empty() {
        root.draw(&Text::new(
            label.to_string(),
            (bar_x, bar_y_top - 22),
            ("sans-serif", 21).into_font(),
        ))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }
    Ok(())
}

// ── 2-D mesh plot (snapshots & model parameters) ─────────────────────────────

fn plot_2d_grid(
    data: &[f32],
    x_coords: &[f32],
    z_coords: &[f32],
    title: &str,
    cmap: fn(f32) -> RGBColor,
    diverging: bool,
    out_png: &Path,
) -> anyhow::Result<()> {
    let (nx, nz) = infer_grid(x_coords, z_coords);
    let nuse = nx * nz;

    let x_min = fmin(x_coords);
    let x_max = fmax(x_coords);
    let z_min = fmin(z_coords);
    let z_max = fmax(z_coords);

    let d = &data[..nuse.min(data.len())];
    let (vmin, vmax) = if diverging {
        let amax = d.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
        let amax = if amax == 0.0 { 1.0 } else { amax };
        (-amax, amax)
    } else {
        (fmin(d), fmax(d))
    };
    let vrange = vmax - vmin;

    // Compute image size so the data canvas respects the model's lx/lz ratio.
    // Fixed pixel overhead (margins + labels + colorbar column):
    //   horizontal: left outer(20) + y_label_area(110) + right margin(200) = 330
    //   vertical:   top outer(20) + caption(~55) + x_label_area(80) + bottom(30) = 185
    const CANVAS_W: u32 = 700;
    const OVERHEAD_W: u32 = 330;
    const OVERHEAD_H: u32 = 185;
    let lx = (x_max - x_min).max(1.0);
    let lz = (z_max - z_min).max(1.0);
    let canvas_h = ((CANVAS_W as f32 * lz / lx) as u32).clamp(150, 1400);
    let img_w = CANVAS_W + OVERHEAD_W;
    let img_h = canvas_h + OVERHEAD_H;

    let root = BitMapBackend::new(out_png, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // Right margin wide enough for colorbar + tick labels
    let plot_area = root.margin(20, 30, 20, 200);

    let dx_step = if nx > 1 { (x_max - x_min) / (nx - 1) as f32 } else { 1.0 };
    let dz_step = if nz > 1 { (z_max - z_min) / (nz - 1) as f32 } else { 1.0 };
    let half_dx = dx_step * 0.5;
    let half_dz = dz_step * 0.5;

    let tick_font  = ("sans-serif", 22).into_font();
    let label_font = ("sans-serif", 26).into_font();

    // y-axis reversed so z=0 is at top (depth convention)
    let mut chart = ChartBuilder::on(&plot_area)
        .caption(title, ("sans-serif", 30))
        .x_label_area_size(80)
        .y_label_area_size(110)
        .build_cartesian_2d(
            (x_min - half_dx)..(x_max + half_dx),
            (z_max + half_dz)..(z_min - half_dz),
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("x  [m]")
        .y_desc("z  [m]")
        .axis_desc_style(label_font)
        .label_style(tick_font)
        .x_label_formatter(&|v| sci_label(*v))
        .y_label_formatter(&|v| sci_label(*v))
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // Draw all cells in one series
    chart
        .draw_series((0..nx).flat_map(|ix| {
            let x0 = x_min + ix as f32 * dx_step;
            (0..nz).filter_map(move |iz| {
                let idx = ix * nz + iz;
                if idx >= d.len() {
                    return None;
                }
                let val = d[idx];
                let t = if vrange == 0.0 {
                    0.5
                } else {
                    ((val - vmin) / vrange).clamp(0.0, 1.0)
                };
                let color = cmap(t);
                let z0 = z_min + iz as f32 * dz_step;
                Some(Rectangle::new(
                    [(x0 - half_dx, z0 - half_dz), (x0 + half_dx, z0 + half_dz)],
                    color.filled(),
                ))
            })
        }))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    // Colorbar
    draw_colorbar(&root, cmap, vmin, vmax, "")?;

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    println!("  wrote {}", out_png.display());
    Ok(())
}

// ── Snapshot plotter ──────────────────────────────────────────────────────────

fn plot_snapshot(
    bin_file: &Path,
    x_coords: &[f32],
    z_coords: &[f32],
    out_png: &Path,
) -> anyhow::Result<()> {
    let v = read_raw_bin(bin_file)?;
    let comp = bin_file
        .file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('_').nth(1))
        .unwrap_or("?");
    let title = format!("snapshot  –  {comp}");
    plot_2d_grid(&v, x_coords, z_coords, &title, seismic, true, out_png)
}

// ── Seismogram trace plotter ──────────────────────────────────────────────────

fn plot_traces(npy_file: &Path, out_png: &Path) -> anyhow::Result<()> {
    let Some((nrec, nt, data)) = read_npy(npy_file)? else {
        return Ok(());
    };
    if nrec == 0 || nt == 0 {
        return Ok(());
    }

    let comp = npy_file
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('_').next())
        .unwrap_or("?");

    const SPACING: f32 = 2.5;
    let y_max = (nrec as f32) * SPACING;

    let img_w = 1100u32;
    let img_h = (500 + nrec as u32 * 20).clamp(500, 1400);

    let root = BitMapBackend::new(out_png, (img_w, img_h)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let title = format!("seismograms  –  {comp}  ({nrec} receivers)");
    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(80)
        .y_label_area_size(5)
        .build_cartesian_2d(0f32..(nt as f32), 0f32..y_max)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    chart
        .configure_mesh()
        .x_desc("time step")
        .axis_desc_style(("sans-serif", 26).into_font())
        .label_style(("sans-serif", 22).into_font())
        .x_label_formatter(&|v| sci_label(*v))
        .y_labels(0)
        .draw()
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    for i in 0..nrec {
        let offset = (nrec - 1 - i) as f32 * SPACING;
        let row = &data[i * nt..(i + 1) * nt];
        let norm = row
            .iter()
            .cloned()
            .map(f32::abs)
            .fold(0.0f32, f32::max)
            .max(1e-10);
        chart
            .draw_series(LineSeries::new(
                row.iter()
                    .enumerate()
                    .map(|(t, &v)| (t as f32, v / norm + offset)),
                BLACK.stroke_width(1),
            ))
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    }

    root.present().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    println!("  wrote {}", out_png.display());
    Ok(())
}

// ── Model directory plotter ───────────────────────────────────────────────────

fn plot_model_dir(model_dir: &Path, label: &str) -> anyhow::Result<()> {
    let xf = model_dir.join("proc000000_x.bin");
    let zf = model_dir.join("proc000000_z.bin");
    if !xf.exists() || !zf.exists() {
        println!(
            "  [skip] no coordinate files in {}, skipping model plot",
            model_dir.display()
        );
        return Ok(());
    }

    let x_coords = read_bin(&xf)?;
    let z_coords = read_bin(&zf)?;

    let mut param_files: Vec<PathBuf> = fs::read_dir(model_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("bin")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("proc000000_") && n != "proc000000_x.bin" && n != "proc000000_z.bin")
                    .unwrap_or(false)
        })
        .collect();
    param_files.sort();

    if param_files.is_empty() {
        println!(
            "  [skip] no parameter files in {}",
            model_dir.display()
        );
        return Ok(());
    }

    let tag = if label.is_empty() {
        format!(" ({})", model_dir.file_name().and_then(|n| n.to_str()).unwrap_or(""))
    } else {
        format!(" ({label})")
    };
    println!(
        "\nPlotting {} model parameter(s) from {} …",
        param_files.len(),
        model_dir.display()
    );

    for pf in &param_files {
        let param = pf
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('_').nth(1))
            .unwrap_or("?");
        let data = read_bin(pf)?;
        let title = format!("model{tag}  –  {param}");
        let out_png = pf.with_extension("png");
        if let Err(e) = plot_2d_grid(&data, &x_coords, &z_coords, &title, viridis, false, &out_png)
        {
            println!("  [warn] {e}");
        }
    }
    Ok(())
}

// ── vs-difference plot (adjoint) ──────────────────────────────────────────────

fn plot_vs_diff(example_dir: &Path, output_dir: &Path) -> anyhow::Result<()> {
    let true_f = example_dir.join("model_true").join("proc000000_vs.bin");
    let init_f = example_dir.join("model_init").join("proc000000_vs.bin");
    if !true_f.exists() || !init_f.exists() {
        println!("  [skip] model_true or model_init vs.bin not found, skipping diff plot");
        return Ok(());
    }
    let xf = example_dir.join("model_init").join("proc000000_x.bin");
    let zf = example_dir.join("model_init").join("proc000000_z.bin");
    if !xf.exists() || !zf.exists() {
        println!("  [skip] no coordinate files for diff plot");
        return Ok(());
    }

    let x_coords = read_bin(&xf)?;
    let z_coords = read_bin(&zf)?;
    let vs_true = read_bin(&true_f)?;
    let vs_init = read_bin(&init_f)?;

    let n = x_coords.len().min(vs_true.len()).min(vs_init.len());
    let diff: Vec<f32> = (0..n).map(|i| vs_init[i] - vs_true[i]).collect();

    let out_png = output_dir.join("vs_diff.png");
    plot_2d_grid(
        &diff,
        &x_coords[..n],
        &z_coords[..n],
        "model_init vs  –  model_true vs",
        seismic,
        true,
        &out_png,
    )?;
    println!("\n  wrote {}", out_png.display());
    Ok(())
}

// ── Config helper ─────────────────────────────────────────────────────────────

fn read_mode(example_dir: &Path) -> String {
    let cfg = example_dir.join("config.ini");
    let Ok(text) = fs::read_to_string(&cfg) else { return String::new() };
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == "mode" {
                return v.trim().to_string();
            }
        }
    }
    String::new()
}

fn find_coord_files(example_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    for sub in &["model", "model_init", "model_true"] {
        let xf = example_dir.join(sub).join("proc000000_x.bin");
        let zf = example_dir.join(sub).join("proc000000_z.bin");
        if xf.exists() && zf.exists() {
            return Some((xf, zf));
        }
    }
    None
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: plot_results <example_dir>");
        std::process::exit(1);
    }

    let example_dir = PathBuf::from(&args[1]).canonicalize()?;
    let output_dir = example_dir.join("output");

    if !output_dir.exists() {
        eprintln!("No output directory found: {}", output_dir.display());
        std::process::exit(1);
    }

    let mode = read_mode(&example_dir);

    // ── Adjoint: plot model directories first ─────────────────────────────
    if mode == "adjoint" {
        for (sub, label) in &[("model_init", "initial"), ("model_true", "true")] {
            let d = example_dir.join(sub);
            if d.is_dir() {
                if let Err(e) = plot_model_dir(&d, label) {
                    println!("  [warn] {e}");
                }
            }
        }
        if let Err(e) = plot_vs_diff(&example_dir, &output_dir) {
            println!("  [warn] {e}");
        }
    }

    // ── Coordinate files (for snapshot plots) ─────────────────────────────
    let (x_coords, z_coords) = match find_coord_files(&example_dir) {
        Some((xf, zf)) => (Some(read_bin(&xf)?), Some(read_bin(&zf)?)),
        None => (None, None),
    };

    // ── Velocity snapshots (proc??????_<comp>.bin) ────────────────────────
    let mut snap_files: Vec<PathBuf> = fs::read_dir(&output_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("bin")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| {
                        n.starts_with("proc")
                            && !n.ends_with("_x.bin")
                            && !n.ends_with("_z.bin")
                    })
                    .unwrap_or(false)
        })
        .collect();
    snap_files.sort();

    if !snap_files.is_empty() {
        println!("\nPlotting {} snapshot(s)…", snap_files.len());
    }
    for bf in &snap_files {
        let out_png = bf.with_extension("png");
        match (&x_coords, &z_coords) {
            (Some(xc), Some(zc)) => {
                if let Err(e) = plot_snapshot(bf, xc, zc, &out_png) {
                    println!("  [warn] could not plot {}: {e}", bf.display());
                }
            }
            _ => println!("  [skip] no coordinate files, skipping {}", bf.display()),
        }
    }

    // ── Seismogram traces (.npy in output/traces/ or output/) ─────────────
    let traces_dir = if output_dir.join("traces").exists() {
        output_dir.join("traces")
    } else {
        output_dir.clone()
    };

    // Skip raw-float component dumps (vx_*, vy_*, vz_*) – no shape info.
    let raw_pat = regex_skip_raw_npy;
    let mut npy_files: Vec<PathBuf> = fs::read_dir(&traces_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some("npy")
                && !raw_pat(p.file_name().and_then(|n| n.to_str()).unwrap_or(""))
        })
        .collect();
    npy_files.sort();

    if !npy_files.is_empty() {
        println!("\nPlotting {} seismogram file(s)…", npy_files.len());
    }
    for nf in &npy_files {
        let out_png = nf.with_extension("png");
        if let Err(e) = plot_traces(nf, &out_png) {
            println!("  [warn] could not plot {}: {e}", nf.display());
        }
    }

    if snap_files.is_empty() && npy_files.is_empty() {
        println!("No plottable output files found.");
    }

    Ok(())
}

/// Returns true for raw-float npy dumps like vx_000000.npy, vy_*, vz_*.
/// These are flat arrays with no shape metadata and are skipped.
fn regex_skip_raw_npy(name: &str) -> bool {
    // matches v[xyz]_<digits>.npy
    let bytes = name.as_bytes();
    if bytes.len() < 4 {
        return false;
    }
    bytes[0] == b'v' && matches!(bytes[1], b'x' | b'y' | b'z') && bytes[2] == b'_'
}
