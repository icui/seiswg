"""Generate a homogeneous micropolar model for the 'spin' example.

Grid   : 200 × 200 nodes,  dx = dz = 1 m  (200 m × 200 m)
Physics: homogeneous micropolar / Cosserat medium
  ρ  = 2700 kg/m³
  λ  = μ  = 8.10 GPa   (elastic Lamé parameters)
  ν  = 1.005 GPa        (micropolar coupling modulus)
  J  = 2700 kg/m³       (micro-inertia density)
  λ_c = 0.775 GPa       (couple-stress moduli)
  μ_c = 0.150 GPa
  ν_c = 0.300 GPa

Run from the repository root:
    python wg/examples/spin/generate_model.py
"""

import os
import numpy as np

# ── Grid ──────────────────────────────────────────────────────────────────
dx = 1.0
dz = 1.0
nx = 200
nz = 200
npt = nx * nz

# ── Homogeneous micropolar medium ─────────────────────────────────────────
model = {
    'rho':      np.full(npt, 2700.0,   dtype='float32'),
    'lambda':   np.full(npt, 8.10e9,   dtype='float32'),
    'mu':       np.full(npt, 8.10e9,   dtype='float32'),
    'nu':       np.full(npt, 1.005e9,  dtype='float32'),
    'j':        np.full(npt, 2700.0,   dtype='float32'),
    'lambda_c': np.full(npt, 7.75e8,   dtype='float32'),
    'mu_c':     np.full(npt, 1.50e8,   dtype='float32'),
    'nu_c':     np.full(npt, 3.00e8,   dtype='float32'),
}

x = np.zeros(npt, dtype='float32')
z = np.zeros(npt, dtype='float32')
for i in range(nx):
    for j in range(nz):
        k = i * nz + j
        x[k] = i * dx
        z[k] = j * dz
model['x'] = x
model['z'] = z

# ── Write binary files ─────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'model')
os.makedirs(out_dir, exist_ok=True)

npt_arr = np.array([npt], dtype='int32')

def write_field(name, data):
    path = os.path.join(out_dir, f'proc000000_{name}.bin')
    with open(path, 'wb') as f:
        npt_arr.tofile(f)
        data.astype('float32').tofile(f)
    print(f'  wrote {path}')

for name, data in model.items():
    write_field(name, data)

print(f'\nModel: {nx}×{nz} grid, dx={dx} m, dz={dz} m')
print(f'npt = {npt}')
vs_classical = (model["mu"][0] / model["rho"][0]) ** 0.5
print(f'Classical S-wave speed ≈ {vs_classical:.1f} m/s')
