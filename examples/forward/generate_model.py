"""Generate a homogeneous SH model for the 'forward' example.

Grid   : 200 × 200 nodes,  dx = dz = 100 m  (20 km × 20 km)
Physics: vp = 3000 m/s,  vs = 1732 m/s,  rho = 2700 kg/m³

Run from the repository root:
    python wg/examples/forward/generate_model.py
"""

import os
import numpy as np

# ── Grid ──────────────────────────────────────────────────────────────────
dx = 100.0        # m
dz = 100.0        # m
nx = 200
nz = 200
npt = nx * nz

# ── Homogeneous medium ───────────────────────────────────────────────────
vp  = np.full(npt, 3000.0,  dtype='float32')
vs  = np.full(npt, 1732.0,  dtype='float32')
rho = np.full(npt, 2700.0,  dtype='float32')
x   = np.zeros(npt, dtype='float32')
z   = np.zeros(npt, dtype='float32')

for i in range(nx):
    for j in range(nz):
        k = i * nz + j
        x[k] = i * dx
        z[k] = j * dz

# ── Write binary files ────────────────────────────────────────────────────
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

write_field('x',   x)
write_field('z',   z)
write_field('vp',  vp)
write_field('vs',  vs)
write_field('rho', rho)

print(f'\nModel: {nx}×{nz} grid, dx={dx} m, dz={dz} m')
print(f'npt = {npt}')
