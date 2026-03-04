#!/usr/bin/env python3
"""plot_results.py  –  visualise seispie-wg output

Usage:
    python scripts/plot_results.py  <example_dir>

Finds all seismogram (.npy) files under <example_dir>/output/traces/ and all
velocity snapshot (.bin) files under <example_dir>/output/, and writes a PNG
alongside each.

The model coordinate files (proc000000_x.bin / z.bin) are searched for in the
example directory subtree so that snapshots can be plotted on a spatial grid.
"""

import sys
import re
from glob import glob
from os import path
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ──────────────────────────────────────────────────────────────────────────
# Binary I/O helpers
# ──────────────────────────────────────────────────────────────────────────

def read_bin(filepath):
    """Read a seispie model binary file: int32 npt header + npt*float32 payload."""
    data = Path(filepath).read_bytes()
    npt = int(np.frombuffer(data[:4], dtype="<i4")[0])
    return np.frombuffer(data[4:4 + npt * 4], dtype="<f4").copy()


def read_raw_bin(filepath):
    """Read a raw float32 snapshot file (no header) written by solver.rs."""
    data = Path(filepath).read_bytes()
    return np.frombuffer(data, dtype="<f4").copy()


def find_coord_files(example_dir):
    """Search common locations for the model coordinate binaries."""
    example_dir = Path(example_dir)
    candidates = [
        example_dir / "model" / "proc000000_x.bin",
        example_dir / "model_init" / "proc000000_x.bin",
        example_dir / "model_true" / "proc000000_x.bin",
    ]
    for xf in candidates:
        zf = xf.parent / "proc000000_z.bin"
        if xf.exists() and zf.exists():
            return xf, zf
    return None, None


# ──────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────

def plot_snapshot(bin_file, x_coords, z_coords, out_png):
    """Plot a single velocity-snapshot .bin file as a 2-D colour image."""
    v = read_raw_bin(bin_file)
    npt = len(v)

    x = x_coords[:npt]
    z = z_coords[:npt]

    # Reconstruct nx, nz from coordinate extents
    lx = x.max() - x.min()
    lz = z.max() - z.min()
    nx = int(round((npt * lx / lz) ** 0.5))
    nz = int(round((npt * lz / lx) ** 0.5))

    # Reshape to 2-D grid (layout: v[i*nz + j], i=x-index, j=z-index)
    nx_actual = min(nx, npt // max(nz, 1))
    nz_actual = npt // max(nx_actual, 1)
    grid = v[: nx_actual * nz_actual].reshape(nx_actual, nz_actual).T  # [nz, nx]

    x_ax = np.linspace(x.min(), x.max(), nx_actual)
    z_ax = np.linspace(z.min(), z.max(), nz_actual)

    # Component name from filename
    m = re.search(r"proc\d+_(.+)\.bin", Path(bin_file).name)
    comp = m.group(1) if m else Path(bin_file).stem

    amax = np.abs(grid).max()
    if amax == 0:
        amax = 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(
        x_ax, z_ax, grid,
        cmap="seismic",
        norm=mcolors.TwoSlopeNorm(vmin=-amax, vcenter=0, vmax=amax),
        shading="auto",
    )
    fig.colorbar(im, ax=ax, label="amplitude")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("z  [m]")
    ax.set_title(f"snapshot  –  {comp}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")


def plot_traces(npy_file, out_png, dt=None):
    """Plot a seismogram array: shape [nrec, nt] or flat [nrec*nt]."""
    raw = np.load(npy_file, allow_pickle=False)

    # Determine shape: numpy .npy with explicit shape is [nrec, nt];
    # raw float32 dumps are flat — we can't know nrec, so skip those.
    if raw.ndim == 1:
        # Could be raw float32 – skip plotting (no shape info)
        return
    traces = raw   # [nrec, nt]

    nrec, nt = traces.shape
    t_ax = np.arange(nt) if dt is None else np.arange(nt) * dt

    # Component name from filename (ux_, uy_, uz_, ry_, yi_, …)
    m = re.search(r"([a-z]+)_\d+\.npy", Path(npy_file).name)
    comp = m.group(1) if m else Path(npy_file).stem

    # Normalise each trace independently so all are visible
    norm = np.abs(traces).max(axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    traces_n = traces / norm

    fig, ax = plt.subplots(figsize=(9, max(3, nrec * 0.5)))
    spacing = 2.5
    for i, tr in enumerate(traces_n):
        offset = (nrec - 1 - i) * spacing
        ax.plot(t_ax, tr + offset, color="black", linewidth=0.7)

    ax.set_xlabel("time step" if dt is None else "time  [s]")
    ax.set_ylabel("receiver  (top = 0)")
    ax.set_yticks([])
    ax.set_title(f"seismograms  –  {comp}  ({nrec} receivers)")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <example_dir>", file=sys.stderr)
        sys.exit(1)

    example_dir = Path(sys.argv[1]).resolve()
    output_dir  = example_dir / "output"

    if not output_dir.exists():
        print(f"No output directory found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Coordinate files (for snapshot plots) ────────────────────────────
    xf, zf = find_coord_files(example_dir)
    x_coords = read_bin(xf) if xf else None
    z_coords = read_bin(zf) if zf else None

    # ── Velocity snapshots  (proc??????_<comp>.bin) ────────────────────
    snap_files = sorted(
        f for f in output_dir.glob("proc*.bin")
        if "_x.bin" not in f.name and "_z.bin" not in f.name
    )
    if snap_files:
        print(f"\nPlotting {len(snap_files)} snapshot(s)…")
    for bf in snap_files:
        out_png = bf.with_suffix(".png")
        if x_coords is not None and z_coords is not None:
            try:
                plot_snapshot(bf, x_coords, z_coords, out_png)
            except Exception as e:
                print(f"  [warn] could not plot {bf.name}: {e}")
        else:
            print(f"  [skip] no coordinate files found, skipping {bf.name}")

    # ── Seismogram traces (.npy in traces/ subdirectory) ──────────────────
    traces_dir = output_dir / "traces"
    if not traces_dir.exists():
        traces_dir = output_dir          # spin example writes directly to output/

    npy_files = sorted(
        f for f in traces_dir.glob("*.npy")
        # skip raw-float dumps (vy_*, vx_*, vz_* are raw; uy_*, ux_*, uz_* are npy)
        if not re.match(r"v[xyz]_\d+\.npy", f.name)
    )
    if npy_files:
        print(f"\nPlotting {len(npy_files)} seismogram file(s)…")
    for nf in npy_files:
        out_png = nf.with_suffix(".png")
        try:
            plot_traces(nf, out_png)
        except Exception as e:
            print(f"  [warn] could not plot {nf.name}: {e}")

    if not snap_files and not npy_files:
        print("No plottable output files found.")


if __name__ == "__main__":
    main()
