"""
example_forward.py
==================
Minimal example: run a forward simulation with the standalone fd2d solver.

Usage
-----
    python example_forward.py

It expects the following files to exist (same layout as seispie examples):
    examples/forward/model_true/proc000000_<param>.bin
    examples/forward/sources.dat
    examples/forward/stations.dat

Output goes to examples/forward/output/.
"""

import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from fd2d import FD2D

# ------------------------------------------------------------------ config ---
# All values that would normally come from config.ini
config = {
    # time integration
    'nt': 4800,
    'dt': 0.06,

    # wave physics  (SH only in this example)
    'sh':  'yes',
    'psv': 'no',

    # absorbing boundaries
    'abs_top':    'no',
    'abs_bottom': 'yes',
    'abs_left':   'yes',
    'abs_right':  'yes',
    'abs_width':  20,
    'abs_alpha':  0.015,

    # execution
    'threads_per_block': 512,
    'combine_sources':   'yes',

    # output
    'save_coordinates': True,
    'save_snapshot':    800,
}

# ------------------------------------------------------------------ paths ----
base = 'examples/forward'
path = {
    'sources':       base + '/sources.dat',
    'stations':      base + '/stations.dat',
    'model_true':    base + '/model_true',
    'output':        base + '/output',
    'output_traces': base + '/output/traces',
}

# ------------------------------------------------------------------ run ------
solver = FD2D(config, path)
solver.import_model(model_true=True)
solver.import_sources()
solver.import_stations()
solver.run_forward()

print('Forward simulation complete.')
print('Traces written to', path['output_traces'])
