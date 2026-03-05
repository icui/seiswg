//! seiswg library – exposes the solver as a WASM module for browser use.
//!
//! Build with:
//!   wasm-pack build --target web --out-dir web/pkg
//!
//! Then include web/index.html which loads web/pkg/seiswg.js.

pub mod config;
pub mod solver;

// ── WASM API ─────────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
mod wasm_api {
    use super::*;
    use solver::Vfs;
    use wasm_bindgen::prelude::*;
    use js_sys::{Float32Array, Object, Reflect, Uint8Array};

    /// Initialise panic hook and console logger (call once at startup).
    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
        let _ = console_log::init_with_level(log::Level::Info);
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

    /// Build a VFS from a JS object whose values are `Uint8Array` model files.
    ///
    /// The JS `model.js` script returns `{ "model/proc000000_rho.bin": Uint8Array, … }`
    /// and this function converts that into the Rust `Vfs` (HashMap<String, Vec<u8>>).
    fn vfs_from_js_object(obj: &Object) -> Result<Vfs, JsValue> {
        let mut vfs = Vfs::new();
        let keys = js_sys::Object::keys(obj);
        for i in 0..keys.length() {
            let key = keys.get(i).as_string()
                .ok_or_else(|| JsValue::from_str("model_files: non-string key"))?;
            let val = Reflect::get(obj, &JsValue::from_str(&key))?;
            let bytes = Uint8Array::from(val).to_vec();
            vfs.insert(key, bytes);
        }
        Ok(vfs)
    }

    // ── Simulation ────────────────────────────────────────────────────────────

    /// Run a simulation entirely in-browser via WebGPU.
    ///
    /// Parameters
    /// ----------
    /// - `config_ini`      – INI text for the main simulation
    /// - `sources_dat`     – sources table text
    /// - `stations_dat`    – stations table text
    /// - `config_true_ini` – (adjoint only) INI text for the model_true forward pass
    /// - `model_files`     – JS object `{ path: Uint8Array }` produced by the
    ///                        user-editable `model.js` script
    ///
    /// Returns a JS object with:
    ///   `{ nrec, nt, dt, traces: { uy: Float32Array, … } }`
    /// or for adjoint:
    ///   `{ npt, kmu: Float32Array }`
    #[wasm_bindgen]
    pub async fn run_simulation(
        config_ini:      &str,
        sources_dat:     &str,
        stations_dat:    &str,
        config_true_ini: Option<String>,
        model_files:     JsValue,
    ) -> Result<JsValue, JsValue> {
        let err = |e: anyhow::Error| JsValue::from_str(&format!("{e:#}"));

        // ── Build VFS from the JS model_files object ──────────────────────
        let model_obj = Object::from(model_files);
        let mut vfs = vfs_from_js_object(&model_obj)?;
        vfs.insert("sources.dat".to_string(),  sources_dat.as_bytes().to_vec());
        vfs.insert("stations.dat".to_string(), stations_dat.as_bytes().to_vec());

        // ── Parse config (needed early to determine workflow) ─────────────
        let cfg = config::load_from_str(config_ini, "").map_err(err)?;

        // ── Adjoint: run forward with model_true first to get obs traces ──
        if cfg.workflow == "adjoint" {
            if let Some(true_ini) = &config_true_ini {
                let cfg_true = config::load_from_str(true_ini, "").map_err(err)?;
                let mut slv_true = solver::Solver::new(cfg_true, &vfs).await.map_err(err)?;
                let obs_vfs = slv_true.run_forward().await.map_err(err)?;
                vfs.extend(obs_vfs);
            }
        }

        // ── Main simulation ───────────────────────────────────────────────
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

