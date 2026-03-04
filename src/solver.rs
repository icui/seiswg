//! GPU solver: WebGPU-backed 2-D finite-difference seismic engine.
//!
//! Corresponds to `seispie/solver/fd2d/fd2d.py` + `solver.py` in the Python
//! code-base, but every kernel runs as a WGSL compute shader on the GPU.

use std::f32::consts::PI;
use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::config::Config;

// ──────────────────────────────────────────────────────────────────────────
// Field-slice IDs  (must match the WGSL constants in fd2d.wgsl)
// ──────────────────────────────────────────────────────────────────────────
const N_FIELDS: usize = 44;

const F_VY: usize = 0;
// const F_UY: usize = 1;
// const F_SXY: usize = 2;
// const F_SZY: usize = 3;
// const F_DSY: usize = 4;
// const F_DVYDX: usize = 5;
// const F_DVYDZ: usize = 6;
// const F_VX: usize = 7;
// const F_VZ: usize = 8;
// const F_UX: usize = 9;
// const F_UZ: usize = 10;
// const F_SXX: usize = 11;
// const F_SZZ: usize = 12;
// const F_SXZ: usize = 13;
// const F_DSX: usize = 14;
// const F_DSZ: usize = 15;
// const F_DVXDX: usize = 16;
// const F_DVXDZ: usize = 17;
// const F_DVZDX: usize = 18;
// const F_DVZDZ: usize = 19;
// spin fields 20-29
// model fields
const F_LAM: usize = 30;
const F_MU: usize = 31;
const F_NU: usize = 32;
const F_J: usize = 33;
const F_LAM_C: usize = 34;
const F_MU_C: usize = 35;
const F_NU_C: usize = 36;
const F_RHO: usize = 37;
// const F_BOUND: usize = 38;
// adjoint fields 39-43

// Dynamic fields to zero between sources (indices 0-29).
const N_DYN_FIELDS: usize = 30;

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
// Obs component indices (matching WGSL OBS_* constants)
// ──────────────────────────────────────────────────────────────────────────
const OBS_X: usize = 0;
const OBS_Y: usize = 1;
const OBS_Z: usize = 2;
const OBS_YI: usize = 3;
// const OBS_YC: usize = 4;
const N_OBS_COMP: usize = 5;

const STF_X: usize = 0;
const STF_Y: usize = 1;
const STF_Z: usize = 2;
const N_STF_COMP: usize = 3;

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

/// Read a binary model parameter file produced by the Python solver.
/// Format: 4-byte i32 npt, followed by npt × f32 values.
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
fn load_text_table(path: &Path) -> Result<Vec<Vec<f64>>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("Cannot read {}", path.display()))?;
    let rows: Vec<Vec<f64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
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
fn write_npy_f32(path: &Path, data: &[f32], nrows: usize, ncols: usize) -> Result<()> {
    // Header dict
    let dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        nrows, ncols
    );
    // Total prefix must be a multiple of 64.  Prefix = 10 (magic+ver+len) + header_len.
    let min_header_len = dict.len() + 1; // +1 for trailing '\n'
    let padded = ((min_header_len + 63) / 64) * 64;
    let header_len = padded; // length field in the prefix
    let mut header = dict.into_bytes();
    // Pad with spaces up to header_len - 1, then '\n'
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
fn write_raw_f32(path: &Path, data: &[f32]) -> Result<()> {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    fs::write(path, bytes)
        .with_context(|| format!("Cannot write {}", path.display()))?;
    Ok(())
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
    pub fn new(config: Config) -> Result<Self> {
        // ── GPU initialisation ───────────────────────────────────────────
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("No WebGPU adapter found"))?;

        log::info!("GPU adapter: {:?}", adapter.get_info());

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("seispie"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

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

        let rho_data = load_model_field(&rho_file)?;
        let npt = rho_data.len() as u32;

        let x_data = load_model_field(&model_dir.join("proc000000_x.bin"))?;
        let z_data = load_model_field(&model_dir.join("proc000000_z.bin"))?;

        // Compute grid dimensions from coordinate extents
        let lx = x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - x_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let lz = z_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - z_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let nx = (((npt as f64) * (lx as f64) / (lz as f64)).sqrt().round()) as u32;
        let nz = (((npt as f64) * (lz as f64) / (lx as f64)).sqrt().round()) as u32;
        let dx = lx / (nx - 1) as f32;
        let dz = lz / (nz - 1) as f32;

        log::info!("Grid: nx={} nz={} npt={} dx={:.1} dz={:.1}", nx, nz, npt, dx, dz);

        // ── Load additional model fields ─────────────────────────────────
        let lam_data = load_model_field(&lam_file)?;
        let mu_data  = load_model_field(&mu_file)?;

        let (nu_data, j_data, lam_c_data, mu_c_data, nu_c_data) = if spin {
            (
                load_model_field(&model_dir.join("proc000000_nu.bin"))?,
                load_model_field(&model_dir.join("proc000000_j.bin"))?,
                load_model_field(&model_dir.join("proc000000_lambda_c.bin"))?,
                load_model_field(&model_dir.join("proc000000_mu_c.bin"))?,
                load_model_field(&model_dir.join("proc000000_nu_c.bin"))?,
            )
        } else {
            let zeros = vec![0.0f32; npt as usize];
            (zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(), zeros)
        };

        // ── Allocate flat fields buffer (N_FIELDS × npt f32) ──────────────
        let _fields_bytes = N_FIELDS * npt as usize * 4;
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
        let src_table = load_text_table(&config.paths.sources)?;
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
        let rec_table = load_text_table(&config.paths.stations)?;
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
            sh:         if sh   { 1 } else { 0 },
            psv:        if psv  { 1 } else { 0 },
            spin:       if spin { 1 } else { 0 },
            abs_left:   if bc.left   { 1 } else { 0 },
            abs_right:  if bc.right  { 1 } else { 0 },
            abs_bottom: if bc.bottom { 1 } else { 0 },
            abs_top:    if bc.top    { 1 } else { 0 },
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

    /// Run all forward simulations and save traces to `output/traces/`.
    pub fn run_forward(&mut self) -> Result<()> {
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
        std::fs::create_dir_all(&trace_dir)?;
        if snap > 0 {
            std::fs::create_dir_all(out_dir)?;
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
            self.clear_wavefields(N_DYN_FIELDS * npt as usize * 4)?;

            // ── Clear obs buffer ──────────────────────────────────────────
            {
                let obs_total = (N_OBS_COMP * nrec as usize * nt as usize * 4) as u64;
                let mut enc = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("clear_obs") });
                enc.clear_buffer(&self.obs_buf, 0, Some(obs_total));
                self.queue.submit([enc.finish()]);
            }

            // ── Time loop ─────────────────────────────────────────────────
            for it in 0..nt {
                // Update params on GPU
                let mut p = self.params;
                p.it   = it;
                p.isrc = isrc;
                let params_size = std::mem::size_of::<Params>() as u64;
                self.queue.write_buffer(
                    &self.params_buf, 0,
                    &{
                        let mut buf = vec![0u8; ((params_size + 255) / 256 * 256) as usize];
                        buf[..params_size as usize].copy_from_slice(bytemuck::bytes_of(&p));
                        buf
                    }
                );

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
                    self.save_snapshot(it, sh, psv, spin, out_dir)?;
                }

                // Periodic poll to prevent driver timeout on long runs
                if it % 200 == 199 {
                    self.device.poll(wgpu::Maintain::Wait);
                }
            }
            self.device.poll(wgpu::Maintain::Wait);

            // ── Save traces ───────────────────────────────────────────────
            let obs = self.readback_obs(nrec, nt)?;
            let idx = isrc_out as usize;

            if sh {
                let comp = Self::slice_obs(&obs, OBS_Y, nrec as usize, nt as usize);
                write_npy_f32(
                    &trace_dir.join(format!("uy_{:06}.npy", idx)),
                    &comp, nrec as usize, nt as usize,
                )?;
                // Also write raw (vy_ naming kept for back-compat with examples)
                write_raw_f32(
                    &trace_dir.join(format!("vy_{:06}.npy", idx)),
                    &comp,
                )?;
            }

            if psv {
                let comp_x = Self::slice_obs(&obs, OBS_X, nrec as usize, nt as usize);
                let comp_z = Self::slice_obs(&obs, OBS_Z, nrec as usize, nt as usize);
                write_npy_f32(
                    &trace_dir.join(format!("ux_{:06}.npy", idx)),
                    &comp_x, nrec as usize, nt as usize,
                )?;
                write_npy_f32(
                    &trace_dir.join(format!("uz_{:06}.npy", idx)),
                    &comp_z, nrec as usize, nt as usize,
                )?;
                write_raw_f32(&trace_dir.join(format!("vx_{:06}.npy", idx)), &comp_x)?;
                write_raw_f32(&trace_dir.join(format!("vz_{:06}.npy", idx)), &comp_z)?;

                if spin {
                    let comp_ry = Self::slice_obs(&obs, OBS_Y, nrec as usize, nt as usize);
                    let comp_yi = Self::slice_obs(&obs, OBS_YI, nrec as usize, nt as usize);
                    write_npy_f32(
                        &trace_dir.join(format!("ry_{:06}.npy", idx)),
                        &comp_ry, nrec as usize, nt as usize,
                    )?;
                    write_npy_f32(
                        &trace_dir.join(format!("yi_{:06}.npy", idx)),
                        &comp_yi, nrec as usize, nt as usize,
                    )?;
                }
            }

            log::info!("  → traces saved to {}", trace_dir.display());
        }
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────

    /// Zero the first `byte_count` bytes of `fields_buf` on the GPU.
    fn clear_wavefields(&self, byte_count: usize) -> Result<()> {
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("clear_fields") });
        enc.clear_buffer(&self.fields_buf, 0, Some(byte_count as u64));
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Save a velocity snapshot (called at `save_snapshot` intervals).
    fn save_snapshot(
        &self,
        it: u32,
        sh: bool, psv: bool, spin: bool,
        out_dir: &Path,
    ) -> Result<()> {
        // Read back the fields we need from GPU
        let npt = self.npt as usize;
        // Use the field at F_VY (0) for SH, F_VX (7) for PSV
        let field_size = (N_FIELDS * npt * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("snap_stage"),
            size:               field_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("snap_copy") });
        enc.copy_buffer_to_buffer(&self.fields_buf, 0, &staging, 0, field_size);
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        if sh {
            let vy = &floats[F_VY * npt..(F_VY + 1) * npt];
            write_raw_f32(&out_dir.join(format!("proc{:06}_vy.bin", it)), vy)?;
        }
        if psv {
            let vx_fid = 7;
            let vz_fid = 8;
            let vx = &floats[vx_fid * npt..(vx_fid + 1) * npt];
            let vz = &floats[vz_fid * npt..(vz_fid + 1) * npt];
            write_raw_f32(&out_dir.join(format!("proc{:06}_vx.bin", it)), vx)?;
            write_raw_f32(&out_dir.join(format!("proc{:06}_vz.bin", it)), vz)?;
            if spin {
                let vy_c_fid = 20;
                let ry = &floats[vy_c_fid * npt..(vy_c_fid + 1) * npt];
                write_raw_f32(&out_dir.join(format!("proc{:06}_ry.bin", it)), ry)?;
            }
        }
        drop(data);
        staging.unmap();
        Ok(())
    }

    /// Read the full obs GPU buffer back to host as Vec<f32>.
    fn readback_obs(&self, nrec: u32, nt: u32) -> Result<Vec<f32>> {
        let obs_size = (N_OBS_COMP * nrec as usize * nt as usize * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("obs_stage"),
            size:               obs_size,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("obs_copy") });
        enc.copy_buffer_to_buffer(&self.obs_buf, 0, &staging, 0, obs_size);
        self.queue.submit([enc.finish()]);
        self.device.poll(wgpu::Maintain::Wait);

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mapped = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let result = floats.to_vec();
        drop(mapped);
        staging.unmap();
        Ok(result)
    }

    /// Extract one obs component from the flat buffer.
    fn slice_obs(obs: &[f32], comp: usize, nrec: usize, nt: usize) -> Vec<f32> {
        let stride = nrec * nt;
        obs[comp * stride..(comp + 1) * stride].to_vec()
    }
}
