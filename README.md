# seiswg

2-D finite-difference seismic solver — **Rust + WebGPU** backend.

This is a GPU-accelerated re-implementation of the Python/CUDA [seispie](https://github.com/icui/seispie) solver.
Every compute kernel runs as a WGSL compute shader via [wgpu](https://github.com/gfx-rs/wgpu),
which targets **Vulkan, Metal, and DX12** from a single code-base.

---

## Project layout

```
seiswg/
├── Cargo.toml
├── src/
│   ├── main.rs        Entry point – parse CLI args, dispatch workflow
│   ├── config.rs      INI config loader → typed Config structs
│   └── solver.rs      WebGPU solver (GPU init, buffers, time loop, I/O)
├── shaders/
│   └── fd2d.wgsl      26 WGSL compute kernels
└── examples/
    ├── forward/       Homogeneous SH forward run (small, ~3 s)
    └── spin/          Homogeneous PSV + micropolar-spin forward run
```

---

## Build

```bash
# requires Rust ≥ 1.75 and a Vulkan / Metal / DX12 capable GPU
cargo build --release
```

The binary is written to `target/release/seiswg`.

---

## Usage

```
seiswg <config.ini>
```

`config.ini` uses the same INI format as the Python seispie configs (see
[examples](#examples) below).

Set `RUST_LOG=seispie_wg=info` to suppress verbose wgpu internal messages:

```bash
RUST_LOG=seispie_wg=info seiswg <config.ini>
```

### Supported workflows

| `[workflow] mode` | Status |
|---|---|
| `forward` | ✅ implemented |
| `adjoint` | ✅ implemented |
| `inversion` | planned |

---

## Config reference

### `[workflow]`

| Key | Values | Description |
|---|---|---|
| `mode` | `forward` | Workflow to run |

### `[solver]`

| Key | Default | Description |
|---|---|---|
| `sh` | `no` | Enable SH wavefield (vy, sxy, szy) |
| `psv` | `yes` | Enable PSV wavefield (vx, vz, sxx, szz, sxz) |
| `spin` | `no` | Enable micropolar/spin coupling |
| `nt` | — | Number of time steps |
| `dt` | — | Time step in seconds |
| `abs_left/right/bottom/top` | `no` | Enable absorbing boundaries on each edge |
| `abs_width` | `20` | Width of absorbing layer in grid points |
| `abs_alpha` | `0.015` | Decay coefficient for Gaussian taper |
| `save_snapshot` | `0` | Save velocity snapshot every N steps (0 = off) |
| `combine_sources` | `no` | Inject all sources simultaneously |
| `threads_per_block` | `128` | (informational; GPU dispatch is always 64 threads/workgroup) |

### `[path]`

| Key | Description |
|---|---|
| `output` | Directory for snapshots and other output |
| `output_traces` | Directory for seismogram output (defaults to `output/traces`) |
| `model_init` | Directory containing binary model files |
| `model_true` | Alternative model directory |
| `sources` | Whitespace-delimited source file (see below) |
| `stations` | Whitespace-delimited station file (see below) |

---

## Input file formats

### Model files

Binary files named `proc000000_<param>.bin` in the model directory.
Format: 4-byte little-endian `int32` giving `npt`, followed by `npt × float32`.

**PSV / SH model** (`model_init` or `model_true`):

| File | Description |
|---|---|
| `proc000000_x.bin` | x-coordinates [m] |
| `proc000000_z.bin` | z-coordinates [m] |
| `proc000000_vp.bin` | P-wave velocity [m/s] |
| `proc000000_vs.bin` | S-wave velocity [m/s] |
| `proc000000_rho.bin` | Density [kg/m³] |

**Micropolar-spin model** (also requires `spin = yes`):

| File | Description |
|---|---|
| `proc000000_lambda.bin` / `proc000000_mu.bin` | Lamé parameters [Pa] |
| `proc000000_nu.bin` | Classical micropolar coupling [Pa] |
| `proc000000_j.bin` | Micro-inertia [kg/m³] |
| `proc000000_lambda_c.bin` / `proc000000_mu_c.bin` / `proc000000_nu_c.bin` | Couple-stress moduli [Pa] |
| `proc000000_rho.bin` | Density [kg/m³] |

### sources.dat

One source per line:

```
x_m  z_m  type  f0_Hz  t0_s  angle_deg  amplitude
```

- `x_m`, `z_m` — source position in metres  
- `type` — unused (set to `0`)  
- `f0_Hz` — Ricker dominant frequency  
- `t0_s` — Ricker delay time  
- `angle_deg` — force azimuth (0 = +x, 90 = +z)  
- `amplitude` — force amplitude

### stations.dat

One station per line:

```
x_m  z_m
```

---

## Output files

Seismograms are saved under `output_traces/` (or `output/traces/`):

| File | Component | Layout |
|---|---|---|
| `uy_NNNNNN.npy` | SH displacement y | NumPy array `[nrec, nt]` |
| `ux_NNNNNN.npy` | PSV displacement x | NumPy array `[nrec, nt]` |
| `uz_NNNNNN.npy` | PSV displacement z | NumPy array `[nrec, nt]` |
| `ry_NNNNNN.npy` | Spin micro-rotation y | NumPy array `[nrec, nt]` |
| `yi_NNNNNN.npy` | Isolated rotation | NumPy array `[nrec, nt]` |

`NNNNNN` is the zero-padded source index.

Velocity snapshots (when `save_snapshot > 0`) are written as raw float32 binary
files `proc{it:06}_vx.bin`, `proc{it:06}_vz.bin`, `proc{it:06}_vy.bin` etc. in
the `output/` directory, in the same format as the model files.

---

## Examples

### `examples/forward` — SH forward simulation

Small homogeneous model (200 × 200 grid, 1 source, 25 receivers).

```bash
# 1. generate binary model files
python examples/forward/generate_model.py

# 2. run
./target/release/seiswg examples/forward/config.ini
```

Traces are written to `examples/forward/output/traces/`.

### `examples/spin` — PSV + micropolar-spin forward simulation

Homogeneous micropolar model (500 × 500 grid, 1 source, 7 receivers).

```bash
python examples/spin/generate_model.py
./target/release/seiswg examples/spin/config.ini
```

Traces include `ux`, `uz`, `ry` (total micro-rotation), and `yi` (isolated rotation).

---

## Tests

Integration tests run the release binary against all three bundled examples and
compare output against golden fixtures stored in `tests/fixtures/`.

```bash
# build release binary first, then run tests sequentially (avoids concurrent
# GPU access on single-GPU machines)
cargo test --release -- --test-threads 1
```

### Updating fixtures

After an intentional solver change, regenerate the reference files:

```bash
./run_example.bash   # re-runs all three examples
cp examples/forward/output/proc000100_vy.bin      tests/fixtures/forward/
cp examples/forward/output/traces/uy_000000.npy   tests/fixtures/forward/
cp examples/spin/output/proc000200_vx.bin         tests/fixtures/spin/
cp examples/spin/output/ry_000000.npy             tests/fixtures/spin/
cp examples/adjoint/output/proc000000_kmu.bin     tests/fixtures/adjoint/
```

---

## Browser / GitHub Pages

The solver can run entirely in-browser via **WebAssembly + WebGPU** — no
server required.  The static app lives in `web/index.html`.

### Build the WASM package

```bash
# install wasm-pack (one-time)
cargo install wasm-pack --locked

# add wasm32 target (one-time)
rustup target add wasm32-unknown-unknown

# compile to WASM and output JS glue into web/pkg/
wasm-pack build --release --target web --out-dir web/pkg
```

> **Homebrew Rust conflict** — if Rust was installed via `brew install rust`,
> Homebrew's `rustc` shadows rustup's and `wasm-pack` will complain about a
> missing sysroot.  Prepend the rustup toolchain to `PATH` for the duration
> of the build:
> ```bash
> export PATH="$(rustup which rustc | xargs dirname):$PATH"
> wasm-pack build --release --target web --out-dir web/pkg
> ```
> To make this permanent, add the `export PATH=…` line to your `~/.zprofile`.

### Serve locally

Browsers block ES-module imports and WASM loading from `file://` URLs, so
you need a local HTTP server:

```bash
# Python (no extra install needed)
python3 -m http.server 8080 --directory web

# or: Node.js
npx serve web
```

Then open <http://localhost:8080> in a browser that supports WebGPU
(Chrome ≥ 113, Edge ≥ 113, or Safari ≥ 18).

### Deploy to GitHub Pages

Push to `main` — the `.github/workflows/pages.yml` workflow builds the WASM
package and deploys `web/` automatically via GitHub Actions.

---

## Physics

The solver implements a 4th-order staggered-grid finite-difference scheme in space
and first-order explicit time stepping:

- **SH** — anti-plane shear: $v_y$, $\sigma_{xy}$, $\sigma_{zy}$
- **PSV** — in-plane: $v_x$, $v_z$, $\sigma_{xx}$, $\sigma_{zz}$, $\sigma_{xz}$
- **Micropolar/spin** — classical PSV coupled to micro-rotation $\phi_y$ via
  couple-stress tensor $\Sigma$; yields three seismogram components:
  total rotation, isolated rotation, and curl rotation

Absorbing boundaries use a Gaussian taper applied multiplicatively to the
velocity fields after each update.

---

## Differences from the Python/CUDA version

| Feature | Python (`seispie`) | Rust (`seiswg`) |
|---|---|---|
| Runtime | CUDA (Numba) | WebGPU (wgpu / Vulkan / Metal) |
| Language | Python 3 | Rust 2021 |
| MPI | yes | not yet |
| Adjoint / inversion | yes | yes |
| Workflow config | identical INI format | identical INI format |
| Model file format | identical `.bin` format | identical `.bin` format |
| Trace output | raw float32 `.npy` | NumPy v1.0 `.npy` + raw float32 |
