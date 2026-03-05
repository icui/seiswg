//! Integration tests for seiswg.
//!
//! Each test runs the release binary against one of the three bundled examples
//! (`forward`, `spin`, `adjoint`) and compares selected output files against
//! golden reference fixtures stored in `tests/fixtures/`.
//!
//! # Running
//! ```
//! cargo test --release -- --test-threads 1
//! ```
//! `--test-threads 1` avoids concurrent GPU access when running on a
//! single-GPU machine. Parallelism across examples is fine on multi-GPU
//! machines; adjust as needed.
//!
//! # Updating fixtures
//! After an intentional solver change that alters the ouput, re-run the
//! examples with `run_example.bash` and copy the new reference files:
//! ```
//! cp examples/forward/output/proc000100_vy.bin          tests/fixtures/forward/
//! cp examples/forward/output/traces/uy_000000.npy       tests/fixtures/forward/
//! cp examples/spin/output/proc000200_vx.bin             tests/fixtures/spin/
//! cp examples/spin/output/ry_000000.npy                 tests/fixtures/spin/
//! cp examples/adjoint/output/proc000000_kmu.bin         tests/fixtures/adjoint/
//! ```

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

// ─────────────────────────────────────────────────────────────────────────────
// Relative-error tolerance used for all comparisons.
//
// GPU floating-point is deterministic for a fixed kernel on a fixed adapter,
// but may differ across GPU generations or driver versions.  1e-4 (100 ppm)
// is conservative enough to absorb those differences while still catching any
// real regression.
// ─────────────────────────────────────────────────────────────────────────────
const REL_TOL: f32 = 1e-4;

// ─────────────────────────────────────────────────────────────────────────────
// Paths
// ─────────────────────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is set to the crate root by cargo while testing.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn binary() -> PathBuf {
    workspace_root().join("target").join("release").join("seiswg")
}

fn fixture_dir(example: &str) -> PathBuf {
    workspace_root().join("tests").join("fixtures").join(example)
}

fn example_dir(example: &str) -> PathBuf {
    workspace_root().join("examples").join(example)
}

// ─────────────────────────────────────────────────────────────────────────────
// Runner
// ─────────────────────────────────────────────────────────────────────────────

/// Run the solver with the given config file and assert it succeeds.
fn run_solver(config_path: &Path) {
    let bin = binary();
    assert!(
        bin.exists(),
        "Binary not found at {}\n  → run `cargo build --release` first",
        bin.display()
    );

    let output = Command::new(&bin)
        .arg(config_path)
        .env("RUST_LOG", "error") // suppress info logs during tests
        .output()
        .unwrap_or_else(|e| panic!("Failed to spawn {}: {}", bin.display(), e));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "seiswg exited with {}\nSTDOUT:\n{}\nSTDERR:\n{}",
            output.status, stdout, stderr
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Read a raw-f32 file (`write_raw_f32` / snapshot `.bin` format).
/// Just a flat sequence of little-endian f32 values, no header.
fn read_raw_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} length {} is not a multiple of 4",
        path.display(),
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Read a NumPy v1 `.npy` file produced by `write_npy_f32`.
/// Returns the payload as a flat `Vec<f32>` regardless of shape.
fn read_npy_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));
    assert!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "Not a NumPy file: {}",
        path.display()
    );
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_start = 10 + header_len;
    assert!(
        bytes.len() >= data_start,
        "Truncated .npy header in {}",
        path.display()
    );
    let payload = &bytes[data_start..];
    assert_eq!(
        payload.len() % 4,
        0,
        ".npy payload not a multiple of 4 in {}",
        path.display()
    );
    payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Comparison
// ─────────────────────────────────────────────────────────────────────────────

/// Max element-wise relative error between two f32 slices.
///
/// Near-zero values are handled by normalising against the peak absolute value
/// of the **reference** array (or falling back to an absolute tolerance when
/// the reference is entirely zero).
fn max_rel_err(got: &[f32], reference: &[f32]) -> f32 {
    assert_eq!(
        got.len(),
        reference.len(),
        "Length mismatch: got {} vs reference {}",
        got.len(),
        reference.len()
    );

    let peak = reference
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);

    // If the reference is all zeros, fall back to absolute comparison.
    let denom = if peak > f32::EPSILON { peak } else { 1.0 };

    got.iter()
        .zip(reference.iter())
        .map(|(&g, &r)| (g - r).abs() / denom)
        .fold(0.0f32, f32::max)
}

/// Assert two arrays agree within `REL_TOL`, printing diagnostics on failure.
fn assert_close(got: &[f32], reference: &[f32], label: &str) {
    let err = max_rel_err(got, reference);
    assert!(
        err <= REL_TOL,
        "{}: max relative error {:.2e} exceeds tolerance {:.2e}\n  \
         (reference peak = {:.6e}, got peak = {:.6e})",
        label,
        err,
        REL_TOL,
        reference.iter().copied().map(f32::abs).fold(0.0f32, f32::max),
        got.iter().copied().map(f32::abs).fold(0.0f32, f32::max),
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

/// **Forward (SH)** — 200×200 homogeneous grid, single combined source.
///
/// Checks:
///  - `output/proc000100_vy.bin`   — first velocity snapshot (raw f32)
///  - `output/traces/uy_000000.npy` — displacement seismograms (NumPy)
#[test]
fn test_forward() {
    let ex = example_dir("forward");
    let config = ex.join("config.ini");
    let out = ex.join("output");

    // Clear previous output so we never compare against stale files.
    let _ = fs::remove_dir_all(&out);

    run_solver(&config);

    // ── snapshot ──
    let snap_got = read_raw_f32(&out.join("proc000100_vy.bin"));
    let snap_ref = read_raw_f32(&fixture_dir("forward").join("proc000100_vy.bin"));
    assert_close(&snap_got, &snap_ref, "forward/proc000100_vy.bin");

    // ── traces ──
    let trace_got = read_npy_f32(&out.join("traces").join("uy_000000.npy"));
    let trace_ref = read_npy_f32(&fixture_dir("forward").join("uy_000000.npy"));
    assert_close(&trace_got, &trace_ref, "forward/traces/uy_000000.npy");
}

/// **Spin (PSV + micropolar)** — 200×200 Cosserat grid, single combined source.
///
/// Checks:
///  - `output/proc000200_vx.bin`  — first PSV velocity snapshot (raw f32)
///  - `output/ry_000000.npy`      — micro-rotation seismograms (NumPy)
#[test]
fn test_spin() {
    let ex = example_dir("spin");
    let config = ex.join("config.ini");
    let out = ex.join("output");

    let _ = fs::remove_dir_all(&out);

    run_solver(&config);

    // ── snapshot ──
    let snap_got = read_raw_f32(&out.join("proc000200_vx.bin"));
    let snap_ref = read_raw_f32(&fixture_dir("spin").join("proc000200_vx.bin"));
    assert_close(&snap_got, &snap_ref, "spin/proc000200_vx.bin");

    // ── rotation seismograms ──
    let ry_got = read_npy_f32(&out.join("ry_000000.npy"));
    let ry_ref = read_npy_f32(&fixture_dir("spin").join("ry_000000.npy"));
    assert_close(&ry_got, &ry_ref, "spin/ry_000000.npy");
}

/// **Adjoint (SH)** — 200×200 checkerboard model; 15 sources × 147 receivers.
///
/// Observed traces (`obs_traces/`) are pre-committed to the repository.
/// Only the adjoint pass is run here; the separate `config_true.ini` forward
/// pass that regenerates them is covered by `run_example.bash`.
///
/// Checks:
///  - `output/proc000000_kmu.bin` — µ sensitivity kernel (raw f32)
#[test]
fn test_adjoint() {
    let ex = example_dir("adjoint");
    let config = ex.join("config.ini");
    let out = ex.join("output");

    // Make sure the required inputs exist before running.
    assert!(
        ex.join("obs_traces").join("uy_000000.npy").exists(),
        "obs_traces/uy_000000.npy not found — run `run_example.bash 1` (adjoint) first \
         or ensure the obs_traces are committed"
    );
    assert!(
        ex.join("model_init").join("proc000000_rho.bin").exists(),
        "model_init/proc000000_rho.bin not found — run `python examples/adjoint/generate_model.py` first"
    );

    let _ = fs::remove_dir_all(&out);

    run_solver(&config);

    // ── µ sensitivity kernel ──
    let kmu_got = read_raw_f32(&out.join("proc000000_kmu.bin"));
    let kmu_ref = read_raw_f32(&fixture_dir("adjoint").join("proc000000_kmu.bin"));
    assert_close(&kmu_got, &kmu_ref, "adjoint/proc000000_kmu.bin");
}
