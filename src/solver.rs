//! GPU solver: WebGPU-backed 2-D finite-difference seismic engine.
//!
//! Corresponds to `seispie/solver/fd2d/fd2d.py` + `solver.py` in the Python
//! code-base, but every kernel runs as a WGSL compute shader on the GPU.

use std::collections::HashMap;
use std::f32::consts::PI;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::Path;

/// Virtual file system: maps virtual path strings to byte contents.
pub type Vfs = HashMap<String, Vec<u8>>;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::Config;

// ──────────────────────────────────────────────────────────────────────────
// Field-slice IDs  (must match the WGSL F_* constants in fd2d.wgsl)
// Each field occupies `npt` consecutive f32 values in the GPU fields buffer.
// ──────────────────────────────────────────────────────────────────────────

const N_FIELDS:     usize = 44;
const N_DYN_FIELDS: usize = 30; // dynamic fields zeroed between sources (indices 0–29)

/// Complete set of GPU field-slice IDs, mirroring the WGSL `F_*` constants.
mod field_ids {
    #![allow(dead_code)]
    // SH wavefield
    pub const F_VY:      usize = 0;
    pub const F_UY:      usize = 1;
    pub const F_SXY:     usize = 2;
    pub const F_SZY:     usize = 3;
    pub const F_DSY:     usize = 4;
    pub const F_DVYDX:   usize = 5;
    pub const F_DVYDZ:   usize = 6;
    // PSV wavefield
    pub const F_VX:      usize = 7;
    pub const F_VZ:      usize = 8;
    pub const F_UX:      usize = 9;
    pub const F_UZ:      usize = 10;
    pub const F_SXX:     usize = 11;
    pub const F_SZZ:     usize = 12;
    pub const F_SXZ:     usize = 13;
    pub const F_DSX:     usize = 14;
    pub const F_DSZ:     usize = 15;
    pub const F_DVXDX:   usize = 16;
    pub const F_DVXDZ:   usize = 17;
    pub const F_DVZDX:   usize = 18;
    pub const F_DVZDZ:   usize = 19;
    // Micropolar (spin) wavefield
    pub const F_VY_C:    usize = 20;
    pub const F_UY_C:    usize = 21;
    pub const F_SYX_C:   usize = 22;
    pub const F_SYY_C:   usize = 23;
    pub const F_SYZ_C:   usize = 24;
    pub const F_DSY_C:   usize = 25;
    pub const F_DVYDX_C: usize = 26; // also: fw-wavefield ∂v_y/∂x during adjoint
    pub const F_DVYDZ_C: usize = 27; // also: fw-wavefield ∂v_y/∂z during adjoint
    pub const F_DUZDX:   usize = 28;
    pub const F_DUXDZ:   usize = 29;
    // Model parameters
    pub const F_LAM:     usize = 30;
    pub const F_MU:      usize = 31;
    pub const F_NU:      usize = 32;
    pub const F_J:       usize = 33;
    pub const F_LAM_C:   usize = 34;
    pub const F_MU_C:    usize = 35;
    pub const F_NU_C:    usize = 36;
    pub const F_RHO:     usize = 37;
    pub const F_BOUND:   usize = 38;
    // Adjoint sensitivity kernels
    pub const F_K_LAM:   usize = 39;
    pub const F_K_MU:    usize = 40;
    pub const F_K_RHO:   usize = 41;
    pub const F_GSUM:    usize = 42;
    pub const F_GTMP:    usize = 43;
}
#[allow(unused_imports)]
use field_ids::*;

// ──────────────────────────────────────────────────────────────────────────
// Params uniform (must match the WGSL `Params` struct, size = 96 bytes)
// ──────────────────────────────────────────────────────────────────────────
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Params {
    nx: u32, nz: u32, nt: u32, npt: u32,         // bytes  0-15
    dx: f32, dz: f32, dt: f32, nsrc: u32,         //       16-31
    nrec: u32, it: u32, isrc: i32, sh: u32,       //       32-47
    psv: u32, spin: u32, abs_left: u32, abs_right: u32, //  48-63
    abs_bottom: u32, abs_top: u32, abs_width: u32, abs_alpha: f32, // 64-79
    _pad: [u32; 4],                                //       80-95
}

// ──────────────────────────────────────────────────────────────────────────
// Source and station descriptors
// ──────────────────────────────────────────────────────────────────────────
struct Source {
    grid_index: u32, // k = i*nz + j
    f0: f32,
    t0: f32,
    ang_deg: f32,
    amp: f32,
}

struct Station {
    grid_index: u32,
}

// ──────────────────────────────────────────────────────────────────────────
// Obs / STF component indices  (must match the WGSL OBS_* / STF_* constants)
// ──────────────────────────────────────────────────────────────────────────

/// Observation-buffer component offsets (each = nrec×nt f32 values).
mod obs_ids {
    #![allow(dead_code)]
    pub const OBS_X:  usize = 0; // u_x (PSV)
    pub const OBS_Y:  usize = 1; // u_y (SH) or u_y^c (spin micro-rotation)
    pub const OBS_Z:  usize = 2; // u_z (PSV)
    pub const OBS_YI: usize = 3; // spin: isolated rotation
    pub const OBS_YC: usize = 4; // spin: curl rotation
    pub const N_OBS_COMP: usize = 5;
}
#[allow(unused_imports)]
use obs_ids::*;

/// Source-time-function buffer component offsets (each = nsrc×nt f32 values).
mod stf_ids {
    #![allow(dead_code)]
    pub const STF_X: usize = 0;
    pub const STF_Y: usize = 1;
    pub const STF_Z: usize = 2;
    pub const N_STF_COMP: usize = 3;
}
#[allow(unused_imports)]
use stf_ids::*;

// ──────────────────────────────────────────────────────────────────────────
// Compute-pipeline bundle
// ──────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct Pipelines {
    // Initialisation
    vps2lm:        wgpu::ComputePipeline,
    set_bound:     wgpu::ComputePipeline,
    // SH step
    div_sy:        wgpu::ComputePipeline,
    stf_dsy:       wgpu::ComputePipeline,
    add_vy:        wgpu::ComputePipeline,
    div_vy:        wgpu::ComputePipeline,
    add_sy_sh:     wgpu::ComputePipeline,
    save_obs_y_sh: wgpu::ComputePipeline,
    // PSV step
    div_sxz:       wgpu::ComputePipeline,
    div_sy_c:      wgpu::ComputePipeline,
    div_sxyz_c:    wgpu::ComputePipeline,
    stf_dsxz:      wgpu::ComputePipeline,
    add_vxz:       wgpu::ComputePipeline,
    div_vxz:       wgpu::ComputePipeline,
    add_sxz:       wgpu::ComputePipeline,
    add_vy_c:      wgpu::ComputePipeline,
    div_vy_c:      wgpu::ComputePipeline,
    add_sy_c:      wgpu::ComputePipeline,
    save_obs_x:    wgpu::ComputePipeline,
    save_obs_z:    wgpu::ComputePipeline,
    // Spin seismograms
    compute_du_grad: wgpu::ComputePipeline,
    save_obs_ry:   wgpu::ComputePipeline,
    // Adjoint kernel
    interaction_muy: wgpu::ComputePipeline,
    // Gaussian smoothing
    gaussian_x:    wgpu::ComputePipeline,
    gaussian_z:    wgpu::ComputePipeline,
    // Adjoint-specific helpers
    adj_dsy:       wgpu::ComputePipeline,
    div_uy:        wgpu::ComputePipeline,
    div_fw:        wgpu::ComputePipeline,
    init_gsum:     wgpu::ComputePipeline,
}

// ──────────────────────────────────────────────────────────────────────────
// Main solver struct
// ──────────────────────────────────────────────────────────────────────────
#[allow(dead_code)]
pub struct Solver {
    device:     wgpu::Device,
    queue:      wgpu::Queue,
    params:     Params,
    params_buf: wgpu::Buffer,
    fields_buf: wgpu::Buffer,
    idata_buf:  wgpu::Buffer,
    obs_buf:    wgpu::Buffer,
    stf_buf:    wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipelines:  Pipelines,
    config:     Config,
    sources:    Vec<Source>,
    stations:   Vec<Station>,
    npt:        u32,
}

// ──────────────────────────────────────────────────────────────────────────
// I/O helpers
// ──────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────
// VFS helpers
// ──────────────────────────────────────────────────────────────────────────

/// Parse model field bytes (4-byte int32 npt header + npt × f32).
fn load_model_field_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 4 {
        anyhow::bail!("Model data too small");
    }
    let npt = i32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let expected = 4 + npt * 4;
    if bytes.len() < expected {
        anyhow::bail!("Model data too short (expected {} bytes, got {})", expected, bytes.len());
    }
    Ok(bytes[4..expected]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Load model field from VFS; fall back to filesystem on native builds.
fn load_model_field_vfs(vfs: &Vfs, path: &Path) -> Result<Vec<f32>> {
    let key = path.to_string_lossy();
    if let Some(b) = vfs.get(key.as_ref()).or_else(|| vfs.get(key.trim_start_matches("./"))) {
        return load_model_field_bytes(b);
    }
    #[cfg(not(target_arch = "wasm32"))]
    return load_model_field(path);
    #[cfg(target_arch = "wasm32")]
    anyhow::bail!("VFS: model file not found: {}", path.display())
}

/// Parse a text table from an in-memory string.
fn load_text_table_str(text: &str) -> Result<Vec<Vec<f64>>> {
    let rows = text
        .lines()
        .filter(|l| { let t = l.trim(); !t.is_empty() && !t.starts_with('#') })
        .map(|l| l.split_whitespace().map(|tok| tok.parse::<f64>().unwrap_or(0.0)).collect())
        .collect();
    Ok(rows)
}

/// Load text table from VFS; fall back to filesystem on native builds.
fn load_text_table_vfs(vfs: &Vfs, path: &Path) -> Result<Vec<Vec<f64>>> {
    let key = path.to_string_lossy();
    if let Some(b) = vfs.get(key.as_ref()).or_else(|| vfs.get(key.trim_start_matches("./"))) {
        return load_text_table_str(std::str::from_utf8(b)?);
    }
    #[cfg(not(target_arch = "wasm32"))]
    return load_text_table(path);
    #[cfg(target_arch = "wasm32")]
    anyhow::bail!("VFS: table file not found: {}", path.display())
}

/// Build a NumPy v1 .npy file in memory (little-endian f32, 2-D).
pub fn make_npy_bytes(data: &[f32], nrows: usize, ncols: usize) -> Vec<u8> {
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        nrows, ncols
    );
    let header_len = ((dict.len() + 1 + 63) / 64) * 64;
    let mut header = dict.into_bytes();
    while header.len() < header_len - 1 { header.push(b' '); }
    header.push(b'\n');
    let mut out = Vec::with_capacity(10 + header_len + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(bytemuck::cast_slice(data));
    out
}

/// Parse a NumPy v1 .npy file from memory (f32 2-D array).
fn load_npy_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        anyhow::bail!("Not a NumPy file");
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_start = 10 + header_len;
    if bytes.len() < data_start || (bytes.len() - data_start) % 4 != 0 {
        anyhow::bail!("Truncated .npy file");
    }
    Ok(bytes[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Read a binary model parameter file produced by the Python solver.
/// Format: 4-byte i32 npt, followed by npt × f32 values.
#[cfg(not(target_arch = "wasm32"))]
fn load_model_field(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)
        .with_context(|| format!("Cannot read model file {}", path.display()))?;
    if bytes.len() < 4 {
        anyhow::bail!("Model file too small: {}", path.display());
    }
    let npt = i32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let expected = 4 + npt * 4;
    if bytes.len() < expected {
        anyhow::bail!(
            "Model file {} too short (expected {} bytes, got {})",
            path.display(), expected, bytes.len()
        );
    }
    let floats: Vec<f32> = bytes[4..expected]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(floats)
}

/// Parse a whitespace-separated text file into a 2-D Vec.
#[cfg(not(target_arch = "wasm32"))]
fn load_text_table(path: &Path) -> Result<Vec<Vec<f64>>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .map(|l| {
            l.split_whitespace()
                .map(|tok| tok.parse::<f64>().unwrap_or(0.0))
                .collect()
        })
        .collect();
    Ok(rows)
}

/// Ricker wavelet at time `t`.
/// Returns (stf_x, stf_y, stf_z).
/// Corresponds to `seispie/solver/source/ricker.py`.
fn ricker(t: f32, f0: f32, t0: f32, ang_deg: f32, amp: f32) -> (f32, f32, f32) {
    let a = (PI * f0).powi(2);
    let dt = t - t0;
    let stf = -amp * (1.0 - 2.0 * a * dt * dt) * (-a * dt * dt).exp();
    let ang = ang_deg.to_radians();
    (stf * ang.sin(), stf, -stf * ang.cos())
}

/// Write a 2-D f32 array as a NumPy .npy file (version 1.0, little-endian f4).
/// shape = [nrows, ncols].
#[cfg(not(target_arch = "wasm32"))]
fn write_npy_f32(path: &Path, data: &[f32], nrows: usize, ncols: usize) -> Result<()> {
    // Header dict
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        nrows, ncols
    );
    // Total prefix must be a multiple of 64.  Prefix = 10 (magic+ver+len) + header_len.
    let min_header_len = dict.len() + 1; // +1 for trailing '\n'
    let header_len = ((min_header_len + 63) / 64) * 64;
    let mut header = dict.into_bytes();
    // Pad with spaces up to header_len – 1, then '\n'
    while header.len() < header_len - 1 {
        header.push(b' ');
    }
    header.push(b'\n');

    let mut file = fs::File::create(path)
        .with_context(|| format!("Cannot create {}", path.display()))?;
    // Magic + version 1.0
    file.write_all(b"\x93NUMPY\x01\x00")?;
    // Header length as u16 little-endian
    let hl = header_len as u16;
    file.write_all(&hl.to_le_bytes())?;
    file.write_all(&header)?;
    // Data
    let bytes: &[u8] = bytemuck::cast_slice(data);
    file.write_all(bytes)?;
    Ok(())
}

/// Write raw f32 binary (compatible with `np.fromfile(..., dtype='float32')`).
#[cfg(not(target_arch = "wasm32"))]
fn write_raw_f32(path: &Path, data: &[f32]) -> Result<()> {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    fs::write(path, bytes)
        .with_context(|| format!("Cannot write {}", path.display()))?;
    Ok(())
}

/// Read a NumPy v1 .npy file containing a 2-D f32 array.
/// Returns the data as a flat Vec<f32>.
#[cfg(not(target_arch = "wasm32"))]
fn load_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;
    // .npy magic: \x93NUMPY (6 bytes), version (2 bytes), header_len as u16 LE (2 bytes)
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        anyhow::bail!("Not a NumPy file: {}", path.display());
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_start = 10 + header_len;
    if bytes.len() < data_start {
        anyhow::bail!("Truncated .npy header in {}", path.display());
    }
    let payload = &bytes[data_start..];
    if payload.len() % 4 != 0 {
        anyhow::bail!("Payload length not a multiple of 4 in {}", path.display());
    }
    let floats: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(floats)
}

/// Compute waveform misfit and adjoint source-time-function.
///
/// `syn`  – synthetic traces, flat [nrec * nt] layout syn[ir*nt + it]
/// `obs`  – observed  traces, same layout
/// Returns (misfit, adstf) where adstf is laid out the same way
/// **but time-reversed**: adstf[ir*nt + (nt-1-it)] = 2*(syn-obs)*taper.
fn compute_waveform_misfit(
    syn: &[f32],
    obs: &[f32],
    nrec: usize,
    nt:   usize,
    dt:   f32,
) -> (f64, Vec<f32>) {
    let mut adstf  = vec![0.0f32; nrec * nt];
    let mut misfit = 0.0f64;

    let t_end       = (nt - 1) as f32 * dt;
    let taper_width = t_end / 10.0;
    let t_min       = taper_width;
    let t_max       = t_end - taper_width;

    for ir in 0..nrec {
        for it in 0..nt {
            let kt  = ir * nt + it;
            let akt = ir * nt + (nt - 1 - it); // time-reversed index

            let t = it as f32 * dt;
            let taper = if t <= t_min {
                0.5 + 0.5 * ((PI * (t_min - t) / taper_width).cos())
            } else if t >= t_max {
                0.5 + 0.5 * ((PI * (t_max - t) / taper_width).cos())
            } else {
                1.0
            };

            let diff = (syn[kt] - obs[kt]) * taper;
            misfit  += (diff as f64).powi(2);
            adstf[akt] = diff * taper * 2.0;
        }
    }
    (misfit.sqrt(), adstf)
}

// ──────────────────────────────────────────────────────────────────────────
// Pipeline builder helpers
// ──────────────────────────────────────────────────────────────────────────

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:               Some(entry),
        layout:              Some(layout),
        module:              shader,
        entry_point:         entry,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache:               None,
    })
}

fn make_all_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> Pipelines {
    let p = |e: &str| make_pipeline(device, layout, shader, e);
    Pipelines {
        vps2lm:          p("vps2lm"),
        set_bound:       p("set_bound"),
        div_sy:          p("div_sy"),
        stf_dsy:         p("stf_dsy"),
        add_vy:          p("add_vy"),
        div_vy:          p("div_vy"),
        add_sy_sh:       p("add_sy_sh"),
        save_obs_y_sh:   p("save_obs_y_sh"),
        div_sxz:         p("div_sxz"),
        div_sy_c:        p("div_sy_c"),
        div_sxyz_c:      p("div_sxyz_c"),
        stf_dsxz:        p("stf_dsxz"),
        add_vxz:         p("add_vxz"),
        div_vxz:         p("div_vxz"),
        add_sxz:         p("add_sxz"),
        add_vy_c:        p("add_vy_c"),
        div_vy_c:        p("div_vy_c"),
        add_sy_c:        p("add_sy_c"),
        save_obs_x:      p("save_obs_x"),
        save_obs_z:      p("save_obs_z"),
        compute_du_grad: p("compute_du_grad"),
        save_obs_ry:     p("save_obs_ry"),
        interaction_muy: p("interaction_muy"),
        gaussian_x:      p("gaussian_x"),
        gaussian_z:      p("gaussian_z"),
        adj_dsy:         p("adj_dsy"),
        div_uy:          p("div_uy"),
        div_fw:          p("div_fw"),
        init_gsum:       p("init_gsum"),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Dispatch helpers  (inline in command pass)
// ──────────────────────────────────────────────────────────────────────────

/// Dispatch a compute pipeline over `ceil(npt / 64)` workgroups.
#[inline]
fn dispatch_npt(
    cpass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    npt: u32,
) {
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, bg, &[]);
    cpass.dispatch_workgroups((npt + 63) / 64, 1, 1);
}

/// Dispatch a compute pipeline over exactly `n` workgroups (one thread each).
#[inline]
fn dispatch_n(
    cpass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    n: u32,
) {
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, bg, &[]);
    cpass.dispatch_workgroups(n, 1, 1);
}

// ──────────────────────────────────────────────────────────────────────────
// Solver::new
// ──────────────────────────────────────────────────────────────────────────
impl Solver {
    /// Number of grid points in the model.
    pub fn npt(&self) -> u32 { self.npt }

    /// Number of grid columns (x direction).
    pub fn nx(&self) -> u32 { self.params.nx }

    /// Number of grid rows (z direction).
    pub fn nz(&self) -> u32 { self.params.nz }

    /// Number of receiver stations.
    pub fn nrec(&self) -> usize { self.stations.len() }

    pub async fn new(config: Config, vfs: &Vfs) -> Result<Self> {
        // ── GPU initialisation ───────────────────────────────────────────
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!("No WebGPU adapter found"))?;;

        log::info!("GPU adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("seispie"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await?;

        // ── Load model ──────────────────────────────────────────────────
        let spin = config.solver.spin;
        let sh   = config.solver.sh;
        let psv  = config.solver.psv;

        let model_dir = config
            .paths
            .model_init
            .as_deref()
            .or(config.paths.model_true.as_deref())
            .ok_or_else(|| anyhow::anyhow!("No model path in config"))?;

        // Determine npt from the first model file
        let (rho_file, lam_file, mu_file) = if spin {
            (
                model_dir.join("proc000000_rho.bin"),
                model_dir.join("proc000000_lambda.bin"),
                model_dir.join("proc000000_mu.bin"),
            )
        } else {
            (
                model_dir.join("proc000000_rho.bin"),
                model_dir.join("proc000000_vp.bin"),
                model_dir.join("proc000000_vs.bin"),
            )
        };

        let rho_data = load_model_field_vfs(vfs, &rho_file)?;
        let npt = rho_data.len() as u32;

        let x_data = load_model_field_vfs(vfs, &model_dir.join("proc000000_x.bin"))?;
        let z_data = load_model_field_vfs(vfs, &model_dir.join("proc000000_z.bin"))?;

        // Compute grid dimensions from coordinate extents
        let extent = |v: &[f32]| {
            v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                - v.iter().cloned().fold(f32::INFINITY, f32::min)
        };
        let lx = extent(&x_data);
        let lz = extent(&z_data);
        let nx = (((npt as f64) * (lx as f64) / (lz as f64)).sqrt().round()) as u32;
        let nz = (((npt as f64) * (lz as f64) / (lx as f64)).sqrt().round()) as u32;
        let dx = lx / (nx - 1) as f32;
        let dz = lz / (nz - 1) as f32;

        log::info!("Grid: nx={} nz={} npt={} dx={:.1} dz={:.1}", nx, nz, npt, dx, dz);

        // ── Load additional model fields ─────────────────────────────────
        let lam_data = load_model_field_vfs(vfs, &lam_file)?;
        let mu_data  = load_model_field_vfs(vfs, &mu_file)?;

        let (nu_data, j_data, lam_c_data, mu_c_data, nu_c_data) = if spin {
            (
                load_model_field_vfs(vfs, &model_dir.join("proc000000_nu.bin"))?,
                load_model_field_vfs(vfs, &model_dir.join("proc000000_j.bin"))?,
                load_model_field_vfs(vfs, &model_dir.join("proc000000_lambda_c.bin"))?,
                load_model_field_vfs(vfs, &model_dir.join("proc000000_mu_c.bin"))?,
                load_model_field_vfs(vfs, &model_dir.join("proc000000_nu_c.bin"))?,
            )
        } else {
            let zeros = vec![0.0f32; npt as usize];
            (zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(), zeros)
        };

        // ── Allocate flat fields buffer (N_FIELDS × npt f32) ──────────────
        let mut fields_init = vec![0.0f32; N_FIELDS * npt as usize];
        // Write model data into their respective slices
        let copy_field = |dst: &mut Vec<f32>, fid: usize, src: &[f32]| {
            let off = fid * npt as usize;
            dst[off..off + src.len()].copy_from_slice(src);
        };
        copy_field(&mut fields_init, F_LAM,   &lam_data);
        copy_field(&mut fields_init, F_MU,    &mu_data);
        copy_field(&mut fields_init, F_NU,    &nu_data);
        copy_field(&mut fields_init, F_J,     &j_data);
        copy_field(&mut fields_init, F_LAM_C, &lam_c_data);
        copy_field(&mut fields_init, F_MU_C,  &mu_c_data);
        copy_field(&mut fields_init, F_NU_C,  &nu_c_data);
        copy_field(&mut fields_init, F_RHO,   &rho_data);

        let fields_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("fields"),
            contents: bytemuck::cast_slice(&fields_init),
            usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
        });
        drop(fields_init);

        // ── Load sources ─────────────────────────────────────────────────
        let src_table = load_text_table_vfs(vfs, &config.paths.sources)?;;
        let nsrc = src_table.len() as u32;
        let nt   = config.solver.grid.nt;
        let dt   = config.solver.grid.dt;

        let mut sources: Vec<Source> = Vec::with_capacity(nsrc as usize);
        for row in &src_table {
            let sx = row.get(0).copied().unwrap_or(0.0) as f32;
            let sz = row.get(1).copied().unwrap_or(0.0) as f32;
            let f0 = row.get(3).copied().unwrap_or(10.0) as f32;
            let t0 = row.get(4).copied().unwrap_or(0.1) as f32;
            let ang = row.get(5).copied().unwrap_or(0.0) as f32;
            let amp = row.get(6).copied().unwrap_or(1.0e10) as f32;
            let ix = (sx / dx).round() as u32;
            let iz = (sz / dz).round() as u32;
            sources.push(Source { grid_index: ix * nz + iz, f0, t0, ang_deg: ang, amp });
        }

        // Build STF arrays: [N_STF_COMP][nsrc * nt] interleaved as flat Vec
        let stf_total = N_STF_COMP * nsrc as usize * nt as usize;
        let mut stf_data = vec![0.0f32; stf_total];
        let stride = nsrc as usize * nt as usize;
        for (isrc, src) in sources.iter().enumerate() {
            for it in 0..nt as usize {
                let t = it as f32 * dt;
                let (sx, sy, sz) = ricker(t, src.f0, src.t0, src.ang_deg, src.amp);
                stf_data[STF_X * stride + isrc * nt as usize + it] = sx;
                stf_data[STF_Y * stride + isrc * nt as usize + it] = sy;
                stf_data[STF_Z * stride + isrc * nt as usize + it] = sz;
            }
        }

        let stf_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("stf"),
            contents: bytemuck::cast_slice(&stf_data),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // ── Load stations ────────────────────────────────────────────────
        let rec_table = load_text_table_vfs(vfs, &config.paths.stations)?;
        let nrec = rec_table.len() as u32;
        let mut stations: Vec<Station> = Vec::with_capacity(nrec as usize);
        for row in &rec_table {
            let rx = row.get(0).copied().unwrap_or(0.0) as f32;
            let rz = row.get(1).copied().unwrap_or(0.0) as f32;
            let ix = (rx / dx).round() as u32;
            let iz = (rz / dz).round() as u32;
            stations.push(Station { grid_index: ix * nz + iz });
        }

        // idata = [src_id(nsrc), rec_id(nrec)]
        let mut idata: Vec<i32> = Vec::with_capacity((nsrc + nrec) as usize);
        idata.extend(sources.iter().map(|s| s.grid_index as i32));
        idata.extend(stations.iter().map(|s| s.grid_index as i32));
        let idata_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("idata"),
            contents: bytemuck::cast_slice(&idata),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // ── Obs buffer ───────────────────────────────────────────────────
        let obs_size = (N_OBS_COMP * nrec as usize * nt as usize * 4) as u64;
        let obs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("obs"),
            size:               obs_size,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_SRC
                              | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Build Params ─────────────────────────────────────────────────
        let bc = &config.solver.boundary;
        let params = Params {
            nx,
            nz,
            nt,
            npt,
            dx,
            dz,
            dt,
            nsrc,
            nrec,
            it:         0,
            isrc:       -1,
            sh:         sh   as u32,
            psv:        psv  as u32,
            spin:       spin as u32,
            abs_left:   bc.left   as u32,
            abs_right:  bc.right  as u32,
            abs_bottom: bc.bottom as u32,
            abs_top:    bc.top    as u32,
            abs_width:  bc.width,
            abs_alpha:  bc.alpha,
            _pad: [0; 4],
        };
        // Round buffer size up to 256 bytes (wgpu uniform offset alignment)
        let params_size = std::mem::size_of::<Params>() as u64;
        let params_buf_size = ((params_size + 255) / 256) * 256;
        let mut params_bytes = vec![0u8; params_buf_size as usize];
        params_bytes[..params_size as usize]
            .copy_from_slice(bytemuck::bytes_of(&params));
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("params"),
            contents: &params_bytes,
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Bind group layout ────────────────────────────────────────────
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("pl"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        // ── Shader module ─────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("fd2d"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/fd2d.wgsl").into()
            ),
        });

        // ── Compile all pipelines ─────────────────────────────────────────
        let pipelines = make_all_pipelines(&device, &pipeline_layout, &shader);

        // ── Bind group ────────────────────────────────────────────────────
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fields_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: idata_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: obs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: stf_buf.as_entire_binding() },
            ],
        });

        // ── Initialise model on GPU ──────────────────────────────────────
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("init"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("init_pass"),
                timestamp_writes: None,
            });
            // Convert vp/vs to Lamé if not spin (spin model already has λ, µ)
            if !spin {
                dispatch_npt(&mut cp, &pipelines.vps2lm, &bind_group, npt);
            }
            // Build absorbing-boundary taper
            dispatch_npt(&mut cp, &pipelines.set_bound, &bind_group, npt);
        }
        queue.submit([enc.finish()]);
        device.poll(wgpu::Maintain::Wait);

        Ok(Solver {
            device,
            queue,
            params,
            params_buf,
            fields_buf,
            idata_buf,
            obs_buf,
            stf_buf,
            bind_group,
            pipelines,
            config,
            sources,
            stations,
            npt,
        })
    }

    // ──────────────────────────────────────────────────────────────────────
    // Run forward simulation
    // ──────────────────────────────────────────────────────────────────────

    /// Run all forward simulations; output trace files are returned as a `Vfs`
    /// (and also written to disk on native builds).
    pub async fn run_forward(&mut self) -> Result<Vfs> {
        let mut output     = Vfs::new();
        let mut snap_bufs: HashMap<String, Vec<f32>> = HashMap::new();
        let mut snap_count = 0u32;
        let combine = self.config.solver.combine_sources;
        let nsrc    = self.sources.len() as u32;
        let sh      = self.config.solver.sh;
        let psv     = self.config.solver.psv;
        let spin    = self.config.solver.spin;
        let snap    = self.config.solver.save_snapshot;
        let nt      = self.params.nt;
        let nrec    = self.params.nrec;
        let npt     = self.npt;

        let out_dir = &self.config.paths.output;
        let trace_dir = self.config.paths.output_traces
            .clone()
            .unwrap_or_else(|| out_dir.join("traces"));
        #[cfg(not(target_arch = "wasm32"))] {
            std::fs::create_dir_all(&trace_dir)?;
            if snap > 0 { std::fs::create_dir_all(out_dir)?; }
        }

        let runs: Vec<(i32, u32)> = if combine {
            vec![(-1, 0)]
        } else {
            (0..nsrc).map(|i| (i as i32, i)).collect()
        };

        for (isrc, isrc_out) in runs {
            log::info!(
                "Forward run  isrc={}  ({} time steps)",
                if isrc < 0 { "all".to_string() } else { isrc.to_string() },
                nt
            );

            // ── Clear dynamic wavefields ──────────────────────────────────
            self.clear_wavefields(N_DYN_FIELDS * npt as usize * 4);

            // ── Clear obs buffer ──────────────────────────────────────────
            self.clear_obs_buffer(nrec, nt);

            // ── Time loop ─────────────────────────────────────────────────
            for it in 0..nt {
                let mut p = self.params;
                p.it   = it;
                p.isrc = isrc;
                self.write_params(&p);

                let mut enc = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("step") });
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label:            Some("step_pass"),
                        timestamp_writes: None,
                    });
                    let bg = &self.bind_group;
                    let pl = &self.pipelines;

                    if sh {
                        dispatch_npt(&mut cp, &pl.div_sy,        bg, npt);
                        dispatch_n  (&mut cp, &pl.stf_dsy,       bg, nsrc);
                        dispatch_npt(&mut cp, &pl.add_vy,        bg, npt);
                        dispatch_npt(&mut cp, &pl.div_vy,        bg, npt);
                        dispatch_npt(&mut cp, &pl.add_sy_sh,     bg, npt);
                        dispatch_n  (&mut cp, &pl.save_obs_y_sh, bg, nrec);
                    }

                    if psv {
                        dispatch_npt(&mut cp, &pl.div_sxz, bg, npt);
                        if spin {
                            dispatch_npt(&mut cp, &pl.div_sy_c,   bg, npt);
                            dispatch_npt(&mut cp, &pl.div_sxyz_c, bg, npt);
                        }
                        dispatch_n  (&mut cp, &pl.stf_dsxz,  bg, nsrc);
                        dispatch_npt(&mut cp, &pl.add_vxz,   bg, npt);
                        dispatch_npt(&mut cp, &pl.div_vxz,   bg, npt);
                        dispatch_npt(&mut cp, &pl.add_sxz,   bg, npt);
                        if spin {
                            dispatch_npt(&mut cp, &pl.add_vy_c, bg, npt);
                            dispatch_npt(&mut cp, &pl.div_vy_c, bg, npt);
                            dispatch_npt(&mut cp, &pl.add_sy_c, bg, npt);
                        }
                        dispatch_n(&mut cp, &pl.save_obs_x, bg, nrec);
                        dispatch_n(&mut cp, &pl.save_obs_z, bg, nrec);
                        if spin {
                            dispatch_npt(&mut cp, &pl.compute_du_grad, bg, npt);
                            dispatch_n  (&mut cp, &pl.save_obs_ry,     bg, nrec);
                        }
                    }
                }
                self.queue.submit([enc.finish()]);

                // Snapshot
                if snap > 0 && it > 0 && it % snap == 0 {
                    self.save_snapshot(it, sh, psv, spin, out_dir, &mut snap_bufs).await?;
                    snap_count += 1;
                }

                // Periodic poll to prevent driver timeout on long runs
                if it % 200 == 199 {
                    self.device.poll(wgpu::Maintain::Wait);
                }
            }
            self.device.poll(wgpu::Maintain::Wait);

            // ── Save traces ───────────────────────────────────────────────
            let obs = self.readback_obs(nrec, nt).await;
            let idx = isrc_out as usize;

            if sh {
                let comp = Self::slice_obs(&obs, OBS_Y, nrec as usize, nt as usize);
                let uy_path = trace_dir.join(format!("uy_{:06}.npy", idx));
                let vy_path = trace_dir.join(format!("vy_{:06}.npy", idx));
                let uy_bytes = make_npy_bytes(&comp, nrec as usize, nt as usize);
                output.insert(uy_path.to_string_lossy().to_string(), uy_bytes.clone());
                output.insert(vy_path.to_string_lossy().to_string(),
                    bytemuck::cast_slice::<f32, u8>(&comp).to_vec());
                #[cfg(not(target_arch = "wasm32"))] {
                    write_npy_f32(&uy_path, &comp, nrec as usize, nt as usize)?;
                    write_raw_f32(&vy_path, &comp)?;
                }
            }

            if psv {
                let comp_x = Self::slice_obs(&obs, OBS_X, nrec as usize, nt as usize);
                let comp_z = Self::slice_obs(&obs, OBS_Z, nrec as usize, nt as usize);
                let ux_path = trace_dir.join(format!("ux_{:06}.npy", idx));
                let uz_path = trace_dir.join(format!("uz_{:06}.npy", idx));
                let vx_path = trace_dir.join(format!("vx_{:06}.npy", idx));
                let vz_path = trace_dir.join(format!("vz_{:06}.npy", idx));
                output.insert(ux_path.to_string_lossy().to_string(),
                    make_npy_bytes(&comp_x, nrec as usize, nt as usize));
                output.insert(uz_path.to_string_lossy().to_string(),
                    make_npy_bytes(&comp_z, nrec as usize, nt as usize));
                output.insert(vx_path.to_string_lossy().to_string(),
                    bytemuck::cast_slice::<f32, u8>(&comp_x).to_vec());
                output.insert(vz_path.to_string_lossy().to_string(),
                    bytemuck::cast_slice::<f32, u8>(&comp_z).to_vec());
                #[cfg(not(target_arch = "wasm32"))] {
                    write_npy_f32(&ux_path, &comp_x, nrec as usize, nt as usize)?;
                    write_npy_f32(&uz_path, &comp_z, nrec as usize, nt as usize)?;
                    write_raw_f32(&vx_path, &comp_x)?;
                    write_raw_f32(&vz_path, &comp_z)?;
                }

                if spin {
                    let comp_ry = Self::slice_obs(&obs, OBS_Y, nrec as usize, nt as usize);
                    let comp_yi = Self::slice_obs(&obs, OBS_YI, nrec as usize, nt as usize);
                    let ry_path = trace_dir.join(format!("ry_{:06}.npy", idx));
                    let yi_path = trace_dir.join(format!("yi_{:06}.npy", idx));
                    output.insert(ry_path.to_string_lossy().to_string(),
                        make_npy_bytes(&comp_ry, nrec as usize, nt as usize));
                    output.insert(yi_path.to_string_lossy().to_string(),
                        make_npy_bytes(&comp_yi, nrec as usize, nt as usize));
                    #[cfg(not(target_arch = "wasm32"))] {
                        write_npy_f32(&ry_path, &comp_ry, nrec as usize, nt as usize)?;
                        write_npy_f32(&yi_path, &comp_yi, nrec as usize, nt as usize)?;
                    }
                }
            }

            log::info!("  → traces saved to {}", trace_dir.display());
        }
        // Encode snapshot frames into output VFS
        if snap_count > 0 {
            output.insert(
                "snapshot/nframes".to_string(),
                snap_count.to_le_bytes().to_vec(),
            );
            for (name, data) in snap_bufs {
                output.insert(
                    format!("snapshot/{}.f32", name),
                    bytemuck::cast_slice::<f32, u8>(&data).to_vec(),
                );
            }
        }
        Ok(output)
    }

    // ──────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────

    /// Zero the first `byte_count` bytes of `fields_buf` on the GPU.
    fn clear_wavefields(&self, byte_count: usize) {
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("clear_fields") });
        enc.clear_buffer(&self.fields_buf, 0, Some(byte_count as u64));
        self.queue.submit([enc.finish()]);
    }

    /// Zero the entire obs buffer on the GPU.
    fn clear_obs_buffer(&self, nrec: u32, nt: u32) {
        let obs_bytes = (N_OBS_COMP * nrec as usize * nt as usize * 4) as u64;
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("clear_obs") });
        enc.clear_buffer(&self.obs_buf, 0, Some(obs_bytes));
        self.queue.submit([enc.finish()]);
    }

    /// Save a velocity snapshot: appends field slices to `snap_bufs` (always)
    /// and writes raw binary files on non-wasm32 builds.
    async fn save_snapshot(
        &self,
        it: u32,
        sh: bool, psv: bool, spin: bool,
        out_dir: &Path,
        snap_bufs: &mut HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        if sh {
            let vy = self.readback_field(F_VY).await;
            snap_bufs.entry("vy".to_string()).or_default().extend_from_slice(&vy);
            #[cfg(not(target_arch = "wasm32"))]
            write_raw_f32(&out_dir.join(format!("proc{:06}_vy.bin", it)), &vy)?;
        }
        if psv {
            let vx = self.readback_field(F_VX).await;
            let vz = self.readback_field(F_VZ).await;
            snap_bufs.entry("vx".to_string()).or_default().extend_from_slice(&vx);
            snap_bufs.entry("vz".to_string()).or_default().extend_from_slice(&vz);
            #[cfg(not(target_arch = "wasm32"))] {
                write_raw_f32(&out_dir.join(format!("proc{:06}_vx.bin", it)), &vx)?;
                write_raw_f32(&out_dir.join(format!("proc{:06}_vz.bin", it)), &vz)?;
            }
            if spin {
                let ry = self.readback_field(F_VY_C).await;
                snap_bufs.entry("ry".to_string()).or_default().extend_from_slice(&ry);
                #[cfg(not(target_arch = "wasm32"))]
                write_raw_f32(&out_dir.join(format!("proc{:06}_ry.bin", it)), &ry)?;
            }
        }
        let _ = out_dir;
        let _ = it;
        Ok(())
    }

    /// Read the full obs GPU buffer back to host as Vec<f32>.
    async fn readback_obs(&self, nrec: u32, nt: u32) -> Vec<f32> {
        let size = (N_OBS_COMP * nrec as usize * nt as usize * 4) as u64;
        self.readback_gpu_buffer(&self.obs_buf, 0, size).await
    }

    /// Extract one obs component from the flat buffer.
    fn slice_obs(obs: &[f32], comp: usize, nrec: usize, nt: usize) -> Vec<f32> {
        let stride = nrec * nt;
        obs[comp * stride..(comp + 1) * stride].to_vec()
    }

    /// Write Params struct to the GPU uniform buffer.
    fn write_params(&self, p: &Params) {
        let params_size = std::mem::size_of::<Params>() as u64;
        let buf_size = ((params_size + 255) / 256 * 256) as usize;
        let mut buf = vec![0u8; buf_size];
        buf[..params_size as usize].copy_from_slice(bytemuck::bytes_of(p));
        self.queue.write_buffer(&self.params_buf, 0, &buf);
    }

    /// Download one field slice (fid) from the GPU fields buffer to a Vec<f32>.
    async fn readback_field(&self, fid: usize) -> Vec<f32> {
        let npt = self.npt as usize;
        let offset = (fid * npt * 4) as u64;
        let size   = (npt * 4) as u64;
        self.readback_gpu_buffer(&self.fields_buf, offset, size).await
    }

    /// Copy `size` bytes from a GPU buffer at `offset` to host memory.
    ///
    /// Uses a oneshot channel so it works on both native and WebGPU/WASM.
    async fn readback_gpu_buffer(&self, src: &wgpu::Buffer, offset: u64, size: u64) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("readback_stage"),
            size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(src, offset, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel::<()>();
        slice.map_async(wgpu::MapMode::Read, move |_| { let _ = tx.send(()); });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.await;

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Upload a slice of f32 values into one field slice (fid) of the GPU fields buffer.
    fn write_field_slice(&self, fid: usize, data: &[f32]) {
        let offset = (fid * self.npt as usize * 4) as u64;
        self.queue.write_buffer(&self.fields_buf, offset, bytemuck::cast_slice(data));
    }

    /// Smooth the `F_K_MU` gradient with a 2-D separable Gaussian.
    ///
    /// `sigma` is in grid points. Uses three ordered compute passes in a single
    /// command encoder: `init_gsum` → `gaussian_x` → `gaussian_z`.
    /// Between passes, WebGPU guarantees storage-buffer visibility.
    fn smooth_gradient(&self, sigma: f32) {
        let npt = self.npt;
        let mut p = self.params;
        p.abs_alpha = sigma; // abs_alpha repurposed as σ for smoothing kernels
        self.write_params(&p);

        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("smooth") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smooth_gsum"), timestamp_writes: None });
            dispatch_npt(&mut cp, &self.pipelines.init_gsum, &self.bind_group, npt);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smooth_x"), timestamp_writes: None });
            dispatch_npt(&mut cp, &self.pipelines.gaussian_x, &self.bind_group, npt);
        }
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smooth_z"), timestamp_writes: None });
            dispatch_npt(&mut cp, &self.pipelines.gaussian_z, &self.bind_group, npt);
        }
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::Maintain::Wait);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Run adjoint simulation
    // ──────────────────────────────────────────────────────────────────────

    /// Run adjoint gradient computation.
    ///
    /// If `combine_sources = no` (default for adjoint):
    ///   Loops over each source independently, accumulating k_µ across all.
    ///   Each source's forward traces are compared against the corresponding
    ///   observed trace file `uy_{isrc:06}.npy`.
    ///
    /// If `combine_sources = yes`:
    ///   All sources fire simultaneously; compares against `uy_000000.npy`.
    ///
    /// Steps per source:
    ///   1. Forward pass (single source), save displacement checkpoints.
    ///   2. Waveform misfit + adjoint STF.
    ///   3. Adjoint time loop, accumulating ∂J/∂µ.
    ///
    /// After all sources:
    ///   4. Gaussian smoothing.
    ///   5. Save gradient as `output/proc000000_kmu.bin`.
    pub async fn run_adjoint(&mut self, vfs: &Vfs) -> Result<Vfs> {
        let mut output = Vfs::new();
        let sh   = self.config.solver.sh;
        let nt   = self.params.nt;
        let nrec = self.params.nrec;
        let npt  = self.npt;
        let nsrc = self.sources.len() as u32;
        let dt   = self.params.dt;
        let adj_interval = self.config.solver.adj_interval;
        let smooth_sigma = self.config.solver.smooth;
        let combine      = self.config.solver.combine_sources;

        // nsa: number of displacement checkpoints saved
        let nsa = nt / adj_interval;
        if nsa == 0 {
            anyhow::bail!("adj_interval ({}) >= nt ({})", adj_interval, nt);
        }

        let out_dir = self.config.paths.output.clone();
        #[cfg(not(target_arch = "wasm32"))]
        std::fs::create_dir_all(&out_dir)?;

        // ── Load observed (reference) traces ─────────────────────────────
        let obs_dir = self.config.paths.traces.clone()
            .ok_or_else(|| anyhow::anyhow!(
                "config [path] traces is required for the adjoint workflow"))?;

        // Helper: load the reference trace for source index i from VFS or disk.
        let load_ref_y = |i: usize| -> Result<Vec<f32>> {
            if sh {
                let fname = if combine {
                    format!("uy_{:06}.npy", 0)
                } else {
                    format!("uy_{:06}.npy", i)
                };
                let p = obs_dir.join(&fname);
                let key = p.to_string_lossy();
                if let Some(b) = vfs.get(key.as_ref())
                    .or_else(|| vfs.get(key.trim_start_matches("./")))
                {
                    return load_npy_bytes(b);
                }
                #[cfg(not(target_arch = "wasm32"))]
                return load_npy_f32(&p)
                    .with_context(|| format!("Cannot load observed traces {}", p.display()));
                #[cfg(target_arch = "wasm32")]
                anyhow::bail!("VFS: obs trace not found: {}", key)
            } else {
                Ok(vec![])
            }
        };

        // ── Clear k_mu accumulator before source loop ─────────────────────
        {
            let kmu_off = (F_K_MU * npt as usize * 4) as u64;
            let kmu_len = (npt as usize * 4) as u64;
            let mut enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("clr_kmu") });
            enc.clear_buffer(&self.fields_buf, kmu_off, Some(kmu_len));
            self.queue.submit([enc.finish()]);
        }

        let src_runs: Vec<(i32, usize)> = if combine {
            vec![(-1i32, 0usize)]
        } else {
            (0..nsrc as usize).map(|i| (i as i32, i)).collect()
        };
        let n_runs = src_runs.len();

        let boundary_alpha = self.params.abs_alpha;
        let ndt = adj_interval as f32 * dt;

        let mut total_misfit = 0.0f64;

        for (run_idx, (isrc, ref_idx)) in src_runs.into_iter().enumerate() {
            log::info!(
                "Adjoint: source {}/{} ({} steps, {} checkpoints)…",
                run_idx + 1, n_runs, nt, nsa
            );

            let ref_y = load_ref_y(ref_idx)?;

            // ── Forward pass ────────────────────────────────────────────
            self.clear_wavefields(N_DYN_FIELDS * npt as usize * 4);
            self.clear_obs_buffer(nrec, nt);

            let mut uy_fwd: Vec<Vec<f32>> = vec![vec![0.0f32; npt as usize]; nsa as usize];

            for it in 0..nt {
                let mut p = self.params;
                p.it        = it;
                p.isrc      = isrc;
                p.abs_alpha = boundary_alpha;
                self.write_params(&p);

                let mut enc = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("fw") });
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("fw_pass"), timestamp_writes: None });
                    let bg = &self.bind_group;
                    let pl = &self.pipelines;
                    if sh {
                        dispatch_npt(&mut cp, &pl.div_sy,        bg, npt);
                        dispatch_n  (&mut cp, &pl.stf_dsy,       bg, nsrc);
                        dispatch_npt(&mut cp, &pl.add_vy,        bg, npt);
                        dispatch_npt(&mut cp, &pl.div_vy,        bg, npt);
                        dispatch_npt(&mut cp, &pl.add_sy_sh,     bg, npt);
                        dispatch_n  (&mut cp, &pl.save_obs_y_sh, bg, nrec);
                    }
                }
                self.queue.submit([enc.finish()]);

                if (it + 1) % adj_interval == 0 {
                    let isa = (nsa - (it + 1) / adj_interval) as usize;
                    uy_fwd[isa] = self.readback_field(F_VY).await;
                }

                if it % 200 == 199 {
                    #[cfg(not(target_arch = "wasm32"))]
                    self.device.poll(wgpu::Maintain::Wait);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            self.device.poll(wgpu::Maintain::Wait);

            // ── Misfit + adjoint STF ────────────────────────────────────
            let syn_obs = self.readback_obs(nrec, nt).await;

            let (misfit, adstf) = if sh {
                let syn_y = Self::slice_obs(&syn_obs, OBS_Y, nrec as usize, nt as usize);
                compute_waveform_misfit(&syn_y, &ref_y, nrec as usize, nt as usize, dt)
            } else {
                (0.0_f64, vec![0.0f32; nrec as usize * nt as usize])
            };
            total_misfit += misfit;
            log::info!("Adjoint: source {} misfit = {:.6e}", run_idx + 1, misfit);

            {
                let offset = (OBS_Y * nrec as usize * nt as usize * 4) as u64;
                self.queue.write_buffer(&self.obs_buf, offset, bytemuck::cast_slice(&adstf));
            }

            // ── Adjoint time loop ───────────────────────────────────────
            self.clear_wavefields(N_DYN_FIELDS * npt as usize * 4);

            for it in 0..nt {
                let mut p = self.params;
                p.it        = it;
                p.isrc      = -1;
                p.abs_alpha = boundary_alpha;
                self.write_params(&p);

                let mut enc = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("adj") });
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("adj_pass"), timestamp_writes: None });
                    let bg = &self.bind_group;
                    let pl = &self.pipelines;
                    if sh {
                        dispatch_npt(&mut cp, &pl.div_sy,    bg, npt);
                        dispatch_n  (&mut cp, &pl.adj_dsy,   bg, nrec);
                        dispatch_npt(&mut cp, &pl.add_vy,    bg, npt);
                        dispatch_npt(&mut cp, &pl.div_vy,    bg, npt);
                        dispatch_npt(&mut cp, &pl.add_sy_sh, bg, npt);
                    }
                }
                self.queue.submit([enc.finish()]);

                if it % adj_interval == 0 {
                    let isae = it / adj_interval;
                    if (isae as usize) < uy_fwd.len() {
                        let fw = uy_fwd[isae as usize].clone();
                        self.write_field_slice(F_DSY, &fw);
                        {
                            let mut p = self.params;
                            p.it        = it;
                            p.abs_alpha = ndt;
                            self.write_params(&p);
                        }
                        let mut enc = self.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: Some("img") });
                        {
                            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("img_pass"), timestamp_writes: None });
                            let bg = &self.bind_group;
                            let pl = &self.pipelines;
                            dispatch_npt(&mut cp, &pl.div_uy,          bg, npt);
                            dispatch_npt(&mut cp, &pl.div_fw,          bg, npt);
                            dispatch_npt(&mut cp, &pl.interaction_muy, bg, npt);
                        }
                        self.queue.submit([enc.finish()]);
                    }
                }

                if it % 200 == 199 {
                    #[cfg(not(target_arch = "wasm32"))]
                    self.device.poll(wgpu::Maintain::Wait);
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            self.device.poll(wgpu::Maintain::Wait);
        }

        log::info!("Adjoint: total misfit = {:.6e}", total_misfit);

        // ── Smooth the gradient ───────────────────────────────────────────
        log::info!("Adjoint: smoothing kernel (sigma={:.1})…", smooth_sigma);
        self.smooth_gradient(smooth_sigma);

        // ── Save gradient ─────────────────────────────────────────────────
        let kmu = self.readback_field(F_K_MU).await;

        // Build model-binary bytes (header + f32 data)
        let mut kmu_model_bytes = Vec::with_capacity(4 + npt as usize * 4);
        kmu_model_bytes.extend_from_slice(&(npt as i32).to_le_bytes());
        kmu_model_bytes.extend_from_slice(bytemuck::cast_slice(&kmu));
        let kmu_path = out_dir.join("proc000000_kmu.bin");
        let kmu_model_path = out_dir.join("gradient_kmu.bin");
        output.insert(kmu_path.to_string_lossy().to_string(),
            bytemuck::cast_slice::<f32, u8>(&kmu).to_vec());
        output.insert(kmu_model_path.to_string_lossy().to_string(), kmu_model_bytes.clone());
        #[cfg(not(target_arch = "wasm32"))] {
            write_raw_f32(&kmu_path, &kmu)?;
            std::fs::write(&kmu_model_path, &kmu_model_bytes)
                .with_context(|| format!("Cannot create {}", kmu_model_path.display()))?;
        }
        log::info!("Adjoint: gradient saved to {}", kmu_path.display());

        Ok(output)
    }
}
