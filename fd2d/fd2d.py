"""
Standalone 2-D staggered-grid finite-difference forward / adjoint solver.

This module is entirely self-contained: the only dependencies are NumPy and
Numba (CUDA).  It does not depend on any seispie workflow infrastructure.

Typical usage
-------------
>>> from fd2d import FD2D
>>> solver = FD2D(config, path)
>>> solver.import_model(model_true=True)
>>> solver.import_sources()
>>> solver.import_stations()
>>> solver.run_forward()               # forward only

Adjoint / sensitivity-kernel run:

>>> solver.import_traces()             # load (or auto-generate) reference traces
>>> k_mu, misfit, mu = solver.compute_gradient()
"""

import math
import numpy as np
from os import path, makedirs
from time import time

from numba import cuda

# Low-level CUDA kernels and time-stepping loops live in solver.py.
# Device-helper functions (idx, idxij) must be imported before the kernels
# in this file that call them are JIT-compiled.
from .solver import (
    idx,
    idxij,
    run_forward as _run_forward,
    run_adjoint as _run_adjoint,
    run_kernel  as _run_kernel,
)

from .ricker import ricker
from .misfit import waveform


# ---------------------------------------------------------------------------
# Additional CUDA kernels (field utilities)
# ---------------------------------------------------------------------------

@cuda.jit
def clear_field(field):
    k = idx()
    if k < field.size:
        field[k] = 0


@cuda.jit
def vps2lm(lam, mu, rho):
    """Convert (vp, vs, rho) → (lambda, mu, rho) in place."""
    k = idx()
    if k < lam.size:
        vp = lam[k]
        vs = mu[k]
        lam[k] = rho[k] * (vp * vp - 2.0 * vs * vs) if vp > vs else 0.0
        mu[k]  = rho[k] * vs * vs


@cuda.jit
def lm2vps(vp, vs, rho):
    """Convert (lambda, mu, rho) → (vp, vs, rho) in place."""
    k = idx()
    if k < vp.size:
        lam = vp[k]
        mu_ = vs[k]
        vp[k] = math.sqrt((lam + 2.0 * mu_) / rho[k])
        vs[k] = math.sqrt(mu_ / rho[k])


@cuda.jit
def set_bound(bound, width, alpha, left, right, bottom, top, nx, nz):
    """Fill the absorbing-boundary taper array."""
    k, i, j = idxij(nz)
    if k < bound.size:
        bound[k] = 1.0
        if left and i + 1 < width:
            aw = alpha * (width - i - 1)
            bound[k] *= math.exp(-aw * aw)
        if right and i > nx - width:
            aw = alpha * (width + i - nx)
            bound[k] *= math.exp(-aw * aw)
        if bottom and j > nz - width:
            aw = alpha * (width + j - nz)
            bound[k] *= math.exp(-aw * aw)
        if top and j + 1 < width:
            aw = alpha * (width - j - 1)
            bound[k] *= math.exp(-aw * aw)


@cuda.jit(device=True)
def _gaussian(x, sigma):
    return (
        1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    ) * math.exp(-x * x / (2.0 * sigma * sigma))


@cuda.jit
def init_gaussian(gsum, sigma, nx, nz):
    k, i, j = idxij(nz)
    if k < nx * nz:
        sumx = 0.0
        for n in range(nx):
            sumx += _gaussian(i - n, sigma)
        sumz = 0.0
        for n in range(nz):
            sumz += _gaussian(j - n, sigma)
        gsum[k] = sumx * sumz


@cuda.jit
def apply_gaussian_x(data, gtmp, sigma, nx, nz):
    k, i, j = idxij(nz)
    if k < nx * nz:
        sumx = 0.0
        for n in range(nx):
            sumx += _gaussian(i - n, sigma) * data[n * nz + j]
        gtmp[k] = sumx


@cuda.jit
def apply_gaussian_z(data, gtmp, gsum, sigma, nx, nz):
    k, i, j = idxij(nz)
    if k < nx * nz:
        sumz = 0.0
        for n in range(nz):
            sumz += _gaussian(j - n, sigma) * gtmp[i * nz + n]
        data[k] = sumz / gsum[k]


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class FD2D:
    """
    Standalone 2-D staggered-grid finite-difference solver (SH / P-SV / spin).

    Parameters
    ----------
    config : dict
        Simulation parameters.  Required keys:

        ====================  ==================================================
        nt                    number of time steps (int)
        dt                    time step in seconds (float)
        sh                    ``'yes'`` / ``'no'``  – enable SH component
        psv                   ``'yes'`` / ``'no'``  – enable P-SV component
        threads_per_block     number of CUDA threads per block (int)
        abs_left              ``'yes'`` / ``'no'``
        abs_right             ``'yes'`` / ``'no'``
        abs_top               ``'yes'`` / ``'no'``
        abs_bottom            ``'yes'`` / ``'no'``
        abs_width             absorbing-boundary width in grid points (int)
        abs_alpha             absorbing-boundary damping coefficient (float)
        ====================  ==================================================

        Optional keys:

        =====================  ================================================
        spin                   ``'yes'`` / ``'no'`` (default ``'no'``)
        combine_sources        ``'yes'`` / ``'no'`` – inject all sources at once
        save_snapshot          snapshot output interval in time steps (int, 0=off)
        save_coordinates       write coordinate files (truthy / falsy)
        adjoint_interval       wavefield-storage interval for adjoint (int)
        smooth                 Gaussian smoothing sigma for kernels (int)
        =====================  ================================================

    path : dict
        File-system paths.  Recognised keys:

        ================  ======================================================
        sources           ASCII source file   (x  z  …  f0  t0  ang  amp)
        stations          ASCII receiver file (x  z)
        model_true        directory with ``proc000000_<param>.bin`` files
        model_init        same layout (needed if *model_true=False*)
        output            output directory
        output_traces     directory for trace output (optional)
        traces            directory with pre-computed trace files (optional)
        ================  ======================================================

    mpi : object, optional
        MPI communicator wrapper exposing ``rank() -> int`` and ``sync()``.
        Leave as ``None`` for single-process runs.
    """

    def __init__(self, config: dict, path_: dict, mpi=None):
        self.config  = config
        self.path    = path_
        self.mpi     = mpi
        self.taskid  = 0

        self.nt   = int(config['nt'])
        self.dt   = float(config['dt'])
        self.sh   = 1 if config.get('sh')   == 'yes' else 0
        self.psv  = 1 if config.get('psv')  == 'yes' else 0
        self.spin = 1 if config.get('spin') == 'yes' else 0

        # populated later by setup_adjoint()
        self.sae = 0
        self.nsa = 0

        self.stream = cuda.stream()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def import_model(self, model_true: bool = True):
        """
        Load model parameters from binary files and allocate GPU arrays.

        Parameters
        ----------
        model_true : bool
            When ``True`` use ``path['model_true']``; otherwise
            ``path['model_init']``.
        """
        model_dir = self.path['model_true'] if model_true else self.path['model_init']

        if self.spin:
            params = ['x', 'z', 'lambda', 'mu', 'nu', 'j',
                      'lambda_c', 'mu_c', 'nu_c', 'rho']
        else:
            params = ['x', 'z', 'vp', 'vs', 'rho']

        model = {}
        for name in params:
            fname = path.join(model_dir, 'proc000000_' + name + '.bin')
            with open(fname) as f:
                if not hasattr(self, 'npt'):
                    f.seek(0)
                    self.npt = np.fromfile(f, dtype='int32', count=1)
                f.seek(4)
                model[name] = np.fromfile(f, dtype='float32')

        npt  = self.npt[0]
        ntpb = int(self.config['threads_per_block'])
        nb   = int(np.ceil(npt / ntpb))
        self.dim = nb, ntpb

        x  = model['x'];  z  = model['z']
        lx = x.max() - x.min()
        lz = z.max() - z.min()
        nx = self.nx = int(np.rint(np.sqrt(npt * lx / lz)))
        nz = self.nz = int(np.rint(np.sqrt(npt * lz / lx)))
        self.dx = lx / (nx - 1)
        self.dz = lz / (nz - 1)

        stream = self.stream
        zeros  = np.zeros(npt, dtype='float32')

        self.rho = cuda.to_device(model['rho'], stream=stream)

        if self.spin:
            self.lam   = cuda.to_device(model['lambda'],   stream=stream)
            self.mu    = cuda.to_device(model['mu'],       stream=stream)
            self.nu    = cuda.to_device(model['nu'],       stream=stream)
            self.j     = cuda.to_device(model['j'],        stream=stream)
            self.lam_c = cuda.to_device(model['lambda_c'], stream=stream)
            self.mu_c  = cuda.to_device(model['mu_c'],     stream=stream)
            self.nu_c  = cuda.to_device(model['nu_c'],     stream=stream)
        else:
            self.lam = cuda.to_device(model['vp'], stream=stream)
            self.mu  = cuda.to_device(model['vs'], stream=stream)
            vps2lm[self.dim](self.lam, self.mu, self.rho)

        # absorbing boundary
        self.bound = cuda.to_device(zeros.copy(), stream=stream)
        abs_left   = 1 if self.config['abs_left']   == 'yes' else 0
        abs_right  = 1 if self.config['abs_right']  == 'yes' else 0
        abs_top    = 1 if self.config['abs_top']    == 'yes' else 0
        abs_bottom = 1 if self.config['abs_bottom'] == 'yes' else 0
        set_bound[self.dim](
            self.bound,
            int(self.config['abs_width']),
            float(self.config['abs_alpha']),
            abs_left, abs_right, abs_bottom, abs_top, nx, nz,
        )

        # wavefield buffers
        wave_arrays = []
        if self.sh:
            wave_arrays += ['vy', 'uy', 'sxy', 'szy', 'dsy', 'dvydx', 'dvydz']
        if self.psv:
            wave_arrays += ['vx', 'vz', 'ux', 'uz', 'sxx', 'szz', 'sxz',
                            'dsx', 'dsz', 'dvxdx', 'dvxdz', 'dvzdx', 'dvzdz']
            if self.spin:
                wave_arrays += ['vy_c', 'uy_c', 'syx_c', 'syy_c', 'syz_c',
                                'dsy_c', 'dvydx_c', 'dvydz_c', 'duzdx', 'duxdz']
        for name in wave_arrays:
            setattr(self, name, cuda.to_device(zeros.copy(), stream=stream))

        # Gaussian smoother
        self.gsum  = cuda.to_device(zeros.copy(), stream=stream)
        self.gtmp  = cuda.to_device(zeros.copy(), stream=stream)
        self.sigma = int(self.config.get('smooth', 1))
        init_gaussian[self.dim](self.gsum, self.sigma, nx, nz)

        if self.config.get('save_coordinates'):
            self.export_field(x, 'x')
            self.export_field(z, 'z')

    # ------------------------------------------------------------------
    # Sources and receivers
    # ------------------------------------------------------------------

    def import_sources(self):
        """
        Load source coordinates and build Ricker source-time functions.

        The ASCII source file must have one source per line:
        ``x  z  [extra…]  f0  t0  ang  amp``.
        """
        src  = np.loadtxt(self.path['sources'], ndmin=2)
        nsrc = self.nsrc = src.shape[0]
        src_id = np.zeros(nsrc, dtype='int32')

        stf_x = np.zeros(nsrc * self.nt, dtype='float32')
        stf_y = np.zeros(nsrc * self.nt, dtype='float32')
        stf_z = np.zeros(nsrc * self.nt, dtype='float32')

        for isrc in range(nsrc):
            sx = int(np.round(src[isrc][0] / self.dx))
            sz = int(np.round(src[isrc][1] / self.dz))
            src_id[isrc] = sx * self.nz + sz
            for it in range(self.nt):
                istf = isrc * self.nt + it
                stf_x[istf], stf_y[istf], stf_z[istf] = ricker(
                    it * self.dt, *src[isrc][3:]
                )

        stream      = self.stream
        self.src_id = cuda.to_device(src_id, stream=stream)
        self.stf_x  = cuda.to_device(stf_x,  stream=stream)
        self.stf_y  = cuda.to_device(stf_y,  stream=stream)
        self.stf_z  = cuda.to_device(stf_z,  stream=stream)

    def import_stations(self):
        """
        Load receiver coordinates and allocate seismogram arrays on the GPU.

        The ASCII station file must have one receiver per line: ``x  z``.
        """
        rec  = np.loadtxt(self.path['stations'], ndmin=2)
        nrec = self.nrec = rec.shape[0]
        rec_id = np.zeros(nrec, dtype='int32')

        for irec in range(nrec):
            rx = int(np.round(rec[irec][0] / self.dx))
            rz = int(np.round(rec[irec][1] / self.dz))
            rec_id[irec] = rx * self.nz + rz

        stream      = self.stream
        blank       = np.zeros(nrec * self.nt, dtype='float32')
        self.rec_id = cuda.to_device(rec_id,       stream=stream)
        self.obs_x  = cuda.to_device(blank.copy(), stream=stream)
        self.obs_y  = cuda.to_device(blank.copy(), stream=stream)
        self.obs_z  = cuda.to_device(blank.copy(), stream=stream)

        if self.spin:
            self.obs_yi = cuda.to_device(blank.copy(), stream=stream)
            self.obs_yc = cuda.to_device(blank.copy(), stream=stream)

    # ------------------------------------------------------------------
    # Forward simulation
    # ------------------------------------------------------------------

    def run_forward(self):
        """
        Run a single forward simulation.

        Uses ``self.taskid`` (default 0) as the source index.  Set
        ``config['combine_sources'] = 'yes'`` to inject all sources at once.

        Synthetic seismograms are stored in ``self.obs_{x,y,z}`` on the GPU.
        If ``path['output_traces']`` is set, traces are also written as
        ``.npy`` files.
        """
        _run_forward(self)

    # ------------------------------------------------------------------
    # Adjoint simulation and kernels
    # ------------------------------------------------------------------

    def setup_adjoint(self):
        """Allocate arrays needed for adjoint simulation and kernel accumulation."""
        self.sae = int(self.config['adjoint_interval'])
        self.nsa = int(self.nt / self.sae)

        stream = self.stream
        npt    = self.nx * self.nz
        blank  = np.zeros(self.nt * self.nrec, dtype='float32')
        zeros  = np.zeros(npt, dtype='float32')

        if self.sh:
            self.adstf_y  = cuda.to_device(blank.copy(), stream=stream)
            self.dvydx_fw = cuda.to_device(zeros.copy(), stream=stream)
            self.dvydz_fw = cuda.to_device(zeros.copy(), stream=stream)
            self.uy_fwd   = np.zeros([self.nsa, npt], dtype='float32')
            self.vy_fwd   = np.zeros([self.nsa, npt], dtype='float32')

        if self.psv:
            self.adstf_x  = cuda.to_device(blank.copy(), stream=stream)
            self.adstf_z  = cuda.to_device(blank.copy(), stream=stream)
            self.dvxdx_fw = cuda.to_device(zeros.copy(), stream=stream)
            self.dvxdz_fw = cuda.to_device(zeros.copy(), stream=stream)
            self.dvzdx_fw = cuda.to_device(zeros.copy(), stream=stream)
            self.dvzdz_fw = cuda.to_device(zeros.copy(), stream=stream)
            self.ux_fwd   = np.zeros([self.nsa, npt], dtype='float32')
            self.vx_fwd   = np.zeros([self.nsa, npt], dtype='float32')
            self.uz_fwd   = np.zeros([self.nsa, npt], dtype='float32')
            self.vz_fwd   = np.zeros([self.nsa, npt], dtype='float32')

        self.k_lam = cuda.to_device(zeros.copy(), stream=stream)
        self.k_mu  = cuda.to_device(zeros.copy(), stream=stream)
        self.k_rho = cuda.to_device(zeros.copy(), stream=stream)

    def run_adjoint(self):
        """
        Run the adjoint simulation.

        Requires ``setup_adjoint()`` to have been called and adjoint STFs
        (``self.adstf_{x,y,z}``) populated by a preceding ``_compute_misfit()``
        call.
        """
        _run_adjoint(self)

    def run_kernel(self, adj: bool = True):
        """
        Run forward + (optionally) adjoint simulations for all sources.

        Call ``import_traces()`` first to make reference traces available.

        Parameters
        ----------
        adj : bool, default True
            When True, also runs adjoint simulations and returns kernels.

        Returns
        -------
        misfit : float
        k_mu   : cuda device array  (only when *adj* is True)
        mu     : cuda device array  (only when *adj* is True)
        """
        return _run_kernel(self, adj)

    # ------------------------------------------------------------------
    # Trace loading / generation
    # ------------------------------------------------------------------

    def import_traces(self):
        """
        Load reference traces from ``path['traces']``, or generate them by
        running forward simulations with the current model.

        Populates ``self.syn_{x,y,z}`` – lists of flat float32 arrays, one
        entry per source.
        """
        nsrc = self.nsrc
        sh   = self.sh
        psv  = self.psv

        self.syn_x = []
        self.syn_y = []
        self.syn_z = []

        if 'traces' in self.path:
            tracedir = self.path['traces']
            for i in range(nsrc):
                if self.mpi and self.mpi.rank() != i:
                    continue
                if sh:
                    self.syn_y.append(
                        np.fromfile('%s/vy_%06d.npy' % (tracedir, i), dtype='float32')
                    )
                if psv:
                    self.syn_x.append(
                        np.fromfile('%s/vx_%06d.npy' % (tracedir, i), dtype='float32')
                    )
                    self.syn_z.append(
                        np.fromfile('%s/vz_%06d.npy' % (tracedir, i), dtype='float32')
                    )
        else:
            stream   = self.stream
            nrec     = self.nrec
            nt       = self.nt
            tracedir = self.path['output'] + '/traces'

            if not self.mpi or self.mpi.rank() == 0:
                print('Generating reference traces …')
                if not path.exists(tracedir):
                    makedirs(tracedir)

            start = time()
            for i in range(nsrc):
                if self.mpi and self.mpi.rank() != i:
                    continue
                if not self.mpi:
                    print('  task %02d / %02d' % (i + 1, nsrc))
                self.taskid = i
                self.run_forward()

                if sh:
                    out = np.zeros(nt * nrec, dtype='float32')
                    self.obs_y.copy_to_host(out, stream=stream)
                    self.syn_y.append(out.copy())
                    out.tofile('%s/vy_%06d.npy' % (tracedir, i))

                if psv:
                    out = np.zeros(nt * nrec, dtype='float32')
                    self.obs_x.copy_to_host(out, stream=stream)
                    self.syn_x.append(out.copy())
                    out.tofile('%s/vx_%06d.npy' % (tracedir, i))

                    out = np.zeros(nt * nrec, dtype='float32')
                    self.obs_z.copy_to_host(out, stream=stream)
                    self.syn_z.append(out.copy())
                    out.tofile('%s/vz_%06d.npy' % (tracedir, i))

                stream.synchronize()

            if not self.mpi:
                print('Elapsed time: %.2fs\n' % (time() - start))

    # ------------------------------------------------------------------
    # Convenience shorthands
    # ------------------------------------------------------------------

    def compute_misfit(self) -> float:
        """Run forward + misfit for all sources (no adjoint)."""
        return self.run_kernel(adj=False)

    def compute_gradient(self):
        """
        Run forward + adjoint for all sources and return host arrays.

        Returns
        -------
        k_mu   : numpy ndarray – shear-modulus sensitivity kernel (smoothed)
        misfit : float
        mu     : numpy ndarray – current shear-modulus model
        """
        misfit, k_mu_dev, mu_dev = self.run_kernel(adj=True)
        npt   = self.nx * self.nz
        out_k = np.zeros(npt, dtype='float32')
        out_m = np.zeros(npt, dtype='float32')
        k_mu_dev.copy_to_host(out_k, stream=self.stream)
        mu_dev.copy_to_host(out_m, stream=self.stream)
        self.stream.synchronize()
        return out_k, misfit, out_m

    def update_model(self, mu: np.ndarray):
        """Replace the current shear-modulus model with *mu* (host array)."""
        self.mu = cuda.to_device(mu, stream=self.stream)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_misfit(self, comp: str, h_syn: np.ndarray) -> float:
        stream = self.stream
        syn    = cuda.to_device(h_syn, stream)
        obs    = getattr(self, 'obs_' + comp)
        adstf  = getattr(self, 'adstf_' + comp)
        return waveform(syn, obs, adstf, self.nt, self.dt, self.nrec, stream)

    def clear_kernels(self):
        dim = self.dim
        clear_field[dim](self.k_lam)
        clear_field[dim](self.k_mu)
        clear_field[dim](self.k_rho)

    def clear_wavefields(self):
        dim = self.dim
        if self.sh:
            for attr in ('vy', 'uy', 'sxy', 'szy'):
                clear_field[dim](getattr(self, attr))
        if self.psv:
            for attr in ('vx', 'vz', 'ux', 'uz', 'sxx', 'szz', 'sxz'):
                clear_field[dim](getattr(self, attr))
            if self.spin:
                for attr in ('vy_c', 'uy_c', 'syx_c', 'syy_c', 'syz_c'):
                    clear_field[dim](getattr(self, attr))

    def smooth(self, data):
        """Apply Gaussian smoothing to GPU array *data* in-place."""
        dim = self.dim
        apply_gaussian_x[dim](data, self.gtmp, self.sigma, self.nx, self.nz)
        apply_gaussian_z[dim](data, self.gtmp, self.gsum, self.sigma, self.nx, self.nz)

    def export_field(self, field: np.ndarray, name: str, it: int = 0):
        """Write *field* to a Fortran-binary file in ``path['output']``."""
        outdir = self.path['output']
        if not path.exists(outdir):
            makedirs(outdir)
        fname = path.join(outdir, 'proc%06d_%s.bin' % (it, name))
        with open(fname, 'w') as f:
            f.seek(0)
            self.npt.tofile(f)
            f.seek(4)
            field.tofile(f)
