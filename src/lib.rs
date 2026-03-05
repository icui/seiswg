//! seiswg library – exposes the solver as a WASM module for browser use.
//!
//! Build with:
//!   wasm-pack build --target web --out-dir web/pkg
//!
//! Then include web/index.html which loads web/pkg/seiswg.js.

pub mod config;
pub mod solver;

#[cfg(target_arch = "wasm32")]
use solver::Vfs;

// ── WASM-only model-generation helpers ──────────────────────────────────────

/// Encode a seispie model binary: int32 npt header + npt × f32 payload.
#[cfg(target_arch = "wasm32")]
fn make_bin(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + data.len() * 4);
    out.extend_from_slice(&(data.len() as i32).to_le_bytes());
    out.extend_from_slice(bytemuck::cast_slice(data));
    out
}

/// Generate in-memory model files for the `forward` example.
#[cfg(target_arch = "wasm32")]
fn gen_forward_vfs() -> Vfs {
    const NX: usize = 200;
    const NZ: usize = 200;
    const DX: f32 = 100.0;
    const DZ: f32 = 100.0;
    const NPT: usize = NX * NZ;
    let mut x = vec![0.0f32; NPT];
    let mut z = vec![0.0f32; NPT];
    for i in 0..NX { for j in 0..NZ {
        x[i * NZ + j] = i as f32 * DX;
        z[i * NZ + j] = j as f32 * DZ;
    }}
    let mut vfs = Vfs::new();
    let pfx = "model/proc000000_";
    vfs.insert(format!("{pfx}x.bin"),   make_bin(&x));
    vfs.insert(format!("{pfx}z.bin"),   make_bin(&z));
    vfs.insert(format!("{pfx}vp.bin"),  make_bin(&vec![3000.0f32; NPT]));
    vfs.insert(format!("{pfx}vs.bin"),  make_bin(&vec![1732.0f32; NPT]));
    vfs.insert(format!("{pfx}rho.bin"), make_bin(&vec![2700.0f32; NPT]));
    vfs
}

/// Generate in-memory model files for the `spin` example.
#[cfg(target_arch = "wasm32")]
fn gen_spin_vfs() -> Vfs {
    const NX: usize = 200;
    const NZ: usize = 200;
    const NPT: usize = NX * NZ;
    let mut x = vec![0.0f32; NPT];
    let mut z = vec![0.0f32; NPT];
    for i in 0..NX { for j in 0..NZ {
        x[i * NZ + j] = i as f32;
        z[i * NZ + j] = j as f32;
    }}
    let mut vfs = Vfs::new();
    let pfx = "model/proc000000_";
    vfs.insert(format!("{pfx}x.bin"),        make_bin(&x));
    vfs.insert(format!("{pfx}z.bin"),        make_bin(&z));
    vfs.insert(format!("{pfx}rho.bin"),      make_bin(&vec![2700.0f32;  NPT]));
    vfs.insert(format!("{pfx}lambda.bin"),   make_bin(&vec![8.10e9f32;  NPT]));
    vfs.insert(format!("{pfx}mu.bin"),       make_bin(&vec![8.10e9f32;  NPT]));
    vfs.insert(format!("{pfx}nu.bin"),       make_bin(&vec![1.005e9f32; NPT]));
    vfs.insert(format!("{pfx}j.bin"),        make_bin(&vec![2700.0f32;  NPT]));
    vfs.insert(format!("{pfx}lambda_c.bin"), make_bin(&vec![7.75e8f32;  NPT]));
    vfs.insert(format!("{pfx}mu_c.bin"),     make_bin(&vec![1.50e8f32;  NPT]));
    vfs.insert(format!("{pfx}nu_c.bin"),     make_bin(&vec![3.00e8f32;  NPT]));
    vfs
}

/// Generate in-memory model files for the `adjoint` example.
#[cfg(target_arch = "wasm32")]
fn gen_adjoint_vfs() -> (Vfs, Vfs) {
    const NX: usize = 201;
    const NZ: usize = 201;
    const DX: f32 = 2400.0;
    const DZ: f32 = 2400.0;
    const NPT: usize = NX * NZ;
    const VP0: f32 = 5500.0;
    const VS0: f32 = 3500.0;
    const RHO0: f32 = 2600.0;
    const CELL: f32 = 40.0;
    const DVS_FRAC: f32 = 0.114;
    let kx = std::f32::consts::PI / CELL;
    let kz = std::f32::consts::PI / CELL;

    let mut x = vec![0.0f32; NPT];
    let mut z = vec![0.0f32; NPT];
    let mut vs_true = vec![0.0f32; NPT];
    for i in 0..NX { for j in 0..NZ {
        let k = i * NZ + j;
        x[k] = i as f32 * DX;
        z[k] = j as f32 * DZ;
        vs_true[k] = VS0 * (1.0 + DVS_FRAC * (kx * i as f32).sin() * (kz * j as f32).sin());
    }}

    let init_pfx = "model_init/proc000000_";
    let mut init_vfs = Vfs::new();
    init_vfs.insert(format!("{init_pfx}x.bin"),   make_bin(&x));
    init_vfs.insert(format!("{init_pfx}z.bin"),   make_bin(&z));
    init_vfs.insert(format!("{init_pfx}vp.bin"),  make_bin(&vec![VP0;  NPT]));
    init_vfs.insert(format!("{init_pfx}vs.bin"),  make_bin(&vec![VS0;  NPT]));
    init_vfs.insert(format!("{init_pfx}rho.bin"), make_bin(&vec![RHO0; NPT]));

    let true_pfx = "model_true/proc000000_";
    let mut true_vfs = Vfs::new();
    true_vfs.insert(format!("{true_pfx}x.bin"),   make_bin(&x));
    true_vfs.insert(format!("{true_pfx}z.bin"),   make_bin(&z));
    true_vfs.insert(format!("{true_pfx}vp.bin"),  make_bin(&vec![VP0;  NPT]));
    true_vfs.insert(format!("{true_pfx}vs.bin"),  make_bin(&vs_true));
    true_vfs.insert(format!("{true_pfx}rho.bin"), make_bin(&vec![RHO0; NPT]));

    (init_vfs, true_vfs)
}

// ── WASM API ─────────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
mod wasm_api {
    use super::*;
    use wasm_bindgen::prelude::*;
    use js_sys::{Float32Array, Object, Reflect, Uint8Array};

    /// Initialise panic hook and console logger (call once at startup).
    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
        // console_log sends log::* output to browser DevTools console
        let _ = console_log::init_with_level(log::Level::Info);
    }

    // ── Helper: build a JS {key: Uint8Array} object from a Vfs ──────────────

    fn vfs_to_js(vfs: &Vfs) -> Object {
        let obj = Object::new();
        for (k, v) in vfs {
            let arr = Uint8Array::from(v.as_slice());
            Reflect::set(&obj, &JsValue::from_str(k), &arr).unwrap();
        }
        obj
    }

    /// Parse a NumPy v1 .npy file (f32, 2-D) from bytes.
    fn parse_npy_f32(bytes: &[u8]) -> Option<Vec<f32>> {
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" { return None; }
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let data_start = 10 + header_len;
        if bytes.len() < data_start { return None; }
        Some(bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    // ── Model generation ─────────────────────────────────────────────────────

    /// Generate model binary files for the specified example.
    ///
    /// Returns a JS object: `{ "model/proc000000_rho.bin": Uint8Array, … }`
    #[wasm_bindgen]
    pub fn gen_model(model_type: &str) -> Result<Object, JsValue> {
        match model_type {
            "forward" => Ok(vfs_to_js(&gen_forward_vfs())),
            "spin"    => Ok(vfs_to_js(&gen_spin_vfs())),
            "adjoint" => {
                let (init, true_) = gen_adjoint_vfs();
                let mut merged = init;
                merged.extend(true_);
                Ok(vfs_to_js(&merged))
            }
            other => Err(JsValue::from_str(&format!("Unknown model type: {other}"))),
        }
    }

    // ── Simulation ────────────────────────────────────────────────────────────

    /// Run a simulation entirely in-browser via WebGPU.
    ///
    /// Parameters
    /// ----------
    /// - `model_type`      – "forward" | "spin" | "adjoint"
    /// - `config_ini`      – INI text for the main simulation
    /// - `sources_dat`     – sources table text
    /// - `stations_dat`    – stations table text
    /// - `config_true_ini` – (adjoint only) INI text for the model_true forward pass
    ///
    /// Returns a JS object with:
    ///   `{ nrec, nt, dt, traces: { uy: Float32Array, … } }`
    /// or for adjoint:
    ///   `{ npt, kmu: Float32Array }`
    #[wasm_bindgen]
    pub async fn run_simulation(
        model_type:      &str,
        config_ini:      &str,
        sources_dat:     &str,
        stations_dat:    &str,
        config_true_ini: Option<String>,
    ) -> Result<JsValue, JsValue> {
        let err = |e: anyhow::Error| JsValue::from_str(&format!("{e:#}"));

        // ── Build base VFS ──────────────────────────────────────────────
        let mut vfs: Vfs = match model_type {
            "forward" => gen_forward_vfs(),
            "spin"    => gen_spin_vfs(),
            "adjoint" => { let (i, t) = gen_adjoint_vfs(); let mut m = i; m.extend(t); m }
            other => return Err(JsValue::from_str(&format!("Unknown model type: {other}"))),
        };
        vfs.insert("sources.dat".to_string(),  sources_dat.as_bytes().to_vec());
        vfs.insert("stations.dat".to_string(), stations_dat.as_bytes().to_vec());

        // ── Adjoint: run forward with model_true first to get obs traces ──
        if model_type == "adjoint" {
            if let Some(true_ini) = &config_true_ini {
                let cfg_true = config::load_from_str(true_ini, "").map_err(err)?;
                let mut slv_true = solver::Solver::new(cfg_true, &vfs).await.map_err(err)?;
                let obs_vfs = slv_true.run_forward().await.map_err(err)?;
                // Merge obs traces into main vfs so adjoint can find them
                vfs.extend(obs_vfs);
            }
        }

        // ── Main simulation ───────────────────────────────────────────────
        let cfg = config::load_from_str(config_ini, "").map_err(err)?;
        let nt  = cfg.solver.grid.nt;
        let dt  = cfg.solver.grid.dt;
        let mut slv = solver::Solver::new(cfg.clone(), &vfs).await.map_err(err)?;

        let result_obj = Object::new();

        match cfg.workflow.as_str() {
            "forward" => {
                let nrec = slv.nrec();
                let npt  = slv.npt();
                let nx   = slv.nx();
                let nz   = slv.nz();
                let out = slv.run_forward().await.map_err(err)?;
                let traces_obj = Object::new();
                for (path, bytes) in &out {
                    if !path.ends_with(".npy") { continue; }
                    let fname = std::path::Path::new(path)
                        .file_stem().and_then(|s| s.to_str()).unwrap_or(path.as_str());
                    if let Some(vals) = parse_npy_f32(bytes) {
                        let arr = Float32Array::from(vals.as_slice());
                        Reflect::set(&traces_obj, &JsValue::from_str(fname), &arr).unwrap();
                    }
                }
                Reflect::set(&result_obj, &JsValue::from_str("traces"), &traces_obj).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("nt"),   &JsValue::from_f64(nt   as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("dt"),   &JsValue::from_f64(dt   as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("nrec"), &JsValue::from_f64(nrec as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("npt"),  &JsValue::from_f64(npt  as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("nx"),   &JsValue::from_f64(nx   as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("nz"),   &JsValue::from_f64(nz   as f64)).unwrap();
                // Wavefield snapshots – keyed "snapshot/<field>.f32" in VFS
                if let Some(nf_b) = out.get("snapshot/nframes") {
                    if let Ok(arr4) = nf_b[..4].try_into() {
                        let snap_nframes = u32::from_le_bytes(arr4);
                        if snap_nframes > 0 {
                            let snaps_obj = Object::new();
                            for (key, bytes) in &out {
                                if let Some(name) = key
                                    .strip_prefix("snapshot/")
                                    .and_then(|k| k.strip_suffix(".f32"))
                                {
                                    let vals: &[f32] = bytemuck::cast_slice(bytes);
                                    let arr = Float32Array::from(vals);
                                    Reflect::set(&snaps_obj, &JsValue::from_str(name), &arr).unwrap();
                                }
                            }
                            Reflect::set(&result_obj, &JsValue::from_str("snapshots"),
                                &snaps_obj).unwrap();
                            Reflect::set(&result_obj, &JsValue::from_str("snap_nframes"),
                                &JsValue::from_f64(snap_nframes as f64)).unwrap();
                        }
                    }
                }
            }
            "adjoint" => {
                let npt = slv.npt();
                let nx  = (npt as f64).sqrt().round() as u32;
                let nz  = (npt + nx - 1) / nx;
                let out = slv.run_adjoint(&vfs).await.map_err(err)?;
                // kmu raw bytes (no npy header) – find by filename
                if let Some(kmu_bytes) = out.iter()
                    .find(|(k, _)| k.ends_with("proc000000_kmu.bin"))
                    .map(|(_, v)| v)
                {
                    let vals: &[f32] = bytemuck::cast_slice(kmu_bytes);
                    let arr = Float32Array::from(vals);
                    Reflect::set(&result_obj, &JsValue::from_str("kmu"), &arr).unwrap();
                }
                Reflect::set(&result_obj, &JsValue::from_str("nt"),  &JsValue::from_f64(nt  as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("dt"),  &JsValue::from_f64(dt  as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("npt"), &JsValue::from_f64(npt as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("nx"),  &JsValue::from_f64(nx  as f64)).unwrap();
                Reflect::set(&result_obj, &JsValue::from_str("nz"),  &JsValue::from_f64(nz  as f64)).unwrap();
            }
            other => return Err(JsValue::from_str(&format!("Unknown workflow: {other}"))),
        }

        Ok(result_obj.into())
    }
}
