#!/usr/bin/env python3
"""Generate model_init/ (homogeneous) and model_true/ (checkerboard vs anomaly).

Can be run from any directory:
    python wg/examples/adjoint/generate_model.py

Matches the Python seispie examples/adjoint setup exactly:
  - 480 km × 480 km domain, 201×201 nodes, dx = dz = 2400 m
  - Background vs = 3500 m/s, vp = 5500 m/s, rho = 2600 kg/m³
  - Checkerboard ±11.4 % vs perturbation (cell = 40 grid pts = 96 km)
  - 12 surface sources spread across x, vertical receiver arrays below each

For model_true/, this script tries to copy the checker directly from the
Python seispie examples/forward/model_true/ (vs.bin, vp.bin, rho.bin).
If those files are not found it falls back to generating an analytical
sin(kx*I)*sin(kz*J) approximation.
"""

import os, struct, shutil
import numpy as np

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Grid ─────────────────────────────────────────────────────────────────────
nx, nz = 201, 201
dx, dz = 2400.0, 2400.0        # m  →  480 km × 480 km domain
npt    = nx * nz

# Index grids (outer product: shape [nx, nz], then ravel)
I = np.outer(np.arange(nx), np.ones(nz, dtype=int))  # x index
J = np.outer(np.ones(nx, dtype=int), np.arange(nz))  # z index

x = (I.ravel() * dx).astype("f4")
z = (J.ravel() * dz).astype("f4")

# ── Background velocities (match Python seispie examples/adjoint) ────────────
vp0  = 5500.0    # m/s
vs0  = 3500.0    # m/s
rho0 = 2600.0    # kg/m³

# ── Checkerboard perturbation (fallback if Python seispie model not present) ──
# Cell side in grid points.  40 grid pts × 2400 m = 96 km per checker cell.
cell     = 40          # grid points per checker cell (96 km)
dvs_frac = 0.114       # ±11.4 % vs perturbation (matches Python seispie)

kx = np.pi / cell
kz = np.pi / cell
checker = (np.sin(kx * I.ravel()) * np.sin(kz * J.ravel())).astype("f4")

vs_true_fallback = (vs0 * (1.0 + dvs_frac * checker)).astype("f4")

# ── I/O helper ───────────────────────────────────────────────────────────────
def write_bin(subdirname, name, arr):
    dirpath = os.path.join(SCRIPT_DIR, subdirname)
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"proc000000_{name}.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(arr)))
        arr.astype("<f4").tofile(f)
    print(f"  wrote {path}")

# ── model_init (homogeneous) ──────────────────────────────────────────────────
print("Writing model_init/ …")
write_bin("model_init", "vp",  np.full(npt, vp0,  dtype="f4"))
write_bin("model_init", "vs",  np.full(npt, vs0,  dtype="f4"))
write_bin("model_init", "rho", np.full(npt, rho0, dtype="f4"))
write_bin("model_init", "x",   x)
write_bin("model_init", "z",   z)

# ── model_true (checkerboard) ─────────────────────────────────────────────────
print("Writing model_true/ …")
# Prefer to copy directly from the Python seispie examples/forward/model_true
# (exact same 201×201 / 2400 m grid, seispie binary format).
py_fwd_model = os.path.join(
    os.path.dirname(SCRIPT_DIR), "..", "..", "examples", "forward", "model_true"
)
py_fwd_model = os.path.normpath(py_fwd_model)
mt_dir = os.path.join(SCRIPT_DIR, "model_true")
os.makedirs(mt_dir, exist_ok=True)

copied = False
if os.path.isdir(py_fwd_model):
    try:
        for fname in ("proc000000_vs.bin", "proc000000_vp.bin", "proc000000_rho.bin"):
            src = os.path.join(py_fwd_model, fname)
            dst = os.path.join(mt_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  copied {src} → {dst}")
        copied = True
    except Exception as e:
        print(f"  WARNING: could not copy from Python seispie model_true: {e}")

if not copied:
    print("  (Python seispie model_true not found, generating analytical approximation)")
    write_bin("model_true", "vp",  np.full(npt, vp0,  dtype="f4"))
    write_bin("model_true", "vs",  vs_true_fallback)
    write_bin("model_true", "rho", np.full(npt, rho0, dtype="f4"))

# Always write coordinate arrays
write_bin("model_true", "x", x)
write_bin("model_true", "z", z)

print()
print(f"Grid: {nx}×{nz}, dx={dx:.0f} m, dz={dz:.0f} m, npt={npt}")
print(f"Domain: {(nx-1)*dx/1000:.0f} km × {(nz-1)*dz/1000:.0f} km")
print(f"Checker cell: {cell} pts = {cell*dx/1000:.0f} km,  ±{dvs_frac*100:.1f} % vs (analytical fallback)")
print(f"Background vs = {vs0:.0f} m/s,  vp = {vp0:.0f} m/s,  rho = {rho0:.0f} kg/m³")
