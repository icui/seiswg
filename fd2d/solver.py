"""
Low-level CUDA kernels for the 2-D finite-difference solver.
All kernels are written with Numba CUDA and are device-independent of
the rest of the seispie workflow.
"""

import numpy as np
from numba import cuda


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def idx():
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    return tx + ty * bw


@cuda.jit(device=True)
def idxij(nz):
    k = idx()
    j = k % nz
    i = int((k - j) / nz)
    return k, i, j


# ---------------------------------------------------------------------------
# spatial derivatives  (4th-order staggered)
# ---------------------------------------------------------------------------

@cuda.jit(device=True)
def diff_x(v, i, k, dx, nx, nz):
    if i >= 2 and i < nx - 2:
        return 9 * (v[k] - v[k - nz]) / (8 * dx) - (v[k + nz] - v[k - 2 * nz]) / (24 * dx)
    return 0


@cuda.jit(device=True)
def diff_x1(v, i, k, dx, nx, nz):
    if i >= 1 and i < nx - 2:
        return 9 * (v[k + nz] - v[k]) / (8 * dx) - (v[k + 2 * nz] - v[k - nz]) / (24 * dx)
    return 0


@cuda.jit(device=True)
def diff_z(v, j, k, dz, nx, nz):
    if j >= 2 and j < nz - 2:
        return 9 * (v[k] - v[k - 1]) / (8 * dz) - (v[k + 1] - v[k - 2]) / (24 * dz)
    return 0


@cuda.jit(device=True)
def diff_z1(v, j, k, dz, nx, nz):
    if j >= 1 and j < nz - 2:
        return 9 * (v[k + 1] - v[k]) / (8 * dz) - (v[k + 2] - v[k - 1]) / (24 * dz)
    return 0


# ---------------------------------------------------------------------------
# stress divergence
# ---------------------------------------------------------------------------

@cuda.jit
def div_sy(dsy, sxy, szy, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsy.size:
        dsy[k] = diff_x(sxy, i, k, dx, nx, nz) + diff_z(szy, j, k, dz, nx, nz)


@cuda.jit
def div_sxz(dsx, dsz, sxx, szz, sxz, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsx.size:
        dsx[k] = diff_x(sxx, i, k, dx, nx, nz) + diff_z(sxz, j, k, dz, nx, nz)
        dsz[k] = diff_x(sxz, i, k, dx, nx, nz) + diff_z(szz, j, k, dz, nx, nz)


@cuda.jit
def div_sxyz_c(dsx, dsz, dsy_c, syy_c, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dsx.size:
        dsx[k] += diff_z(syy_c, j, k, dz, nx, nz)
        dsz[k] -= diff_x(syy_c, i, k, dx, nx, nz)
        dsy_c[k] += 2 * syy_c[k]


# ---------------------------------------------------------------------------
# source injection
# ---------------------------------------------------------------------------

@cuda.jit
def stf_dsy(dsy, stf_y, src_id, isrc, it, nt):
    ib = cuda.blockIdx.x
    if isrc < 0 or isrc == ib:
        ks = ib * nt + it
        km = src_id[ib]
        dsy[km] += stf_y[ks]


@cuda.jit
def stf_dsxz(dsx, dsz, stf_x, stf_z, src_id, isrc, it, nt):
    ib = cuda.blockIdx.x
    if isrc < 0 or isrc == ib:
        ks = ib * nt + it
        km = src_id[ib]
        dsx[km] += stf_x[ks]
        dsz[km] += stf_z[ks]


# ---------------------------------------------------------------------------
# velocity / displacement update
# ---------------------------------------------------------------------------

@cuda.jit
def add_vy(vy, uy, dsy, rho, bound, dt, npt):
    k = idx()
    if k < npt:
        vy[k] = bound[k] * (vy[k] + dt * dsy[k] / rho[k])
        uy[k] += vy[k] * dt


@cuda.jit
def add_vxz(vx, vz, ux, uz, dsx, dsz, rho, bound, dt):
    k = idx()
    if k < vx.size:
        vx[k] = bound[k] * (vx[k] + dt * dsx[k] / rho[k])
        vz[k] = bound[k] * (vz[k] + dt * dsz[k] / rho[k])
        ux[k] += vx[k] * dt
        uz[k] += vz[k] * dt


# ---------------------------------------------------------------------------
# gradient of velocity fields
# ---------------------------------------------------------------------------

@cuda.jit
def div_vy(dvydx, dvydz, vy, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < nx * nz:
        dvydx[k] = diff_x1(vy, i, k, dx, nx, nz)
        dvydz[k] = diff_z1(vy, j, k, dz, nx, nz)


@cuda.jit
def div_vxz(dvxdx, dvxdz, dvzdx, dvzdz, vx, vz, dx, dz, nx, nz):
    k, i, j = idxij(nz)
    if k < dvxdx.size:
        dvxdx[k] = diff_x1(vx, i, k, dx, nx, nz)
        dvxdz[k] = diff_z1(vx, j, k, dz, nx, nz)
        dvzdx[k] = diff_x1(vz, i, k, dx, nx, nz)
        dvzdz[k] = diff_z1(vz, j, k, dz, nx, nz)


# ---------------------------------------------------------------------------
# stress update
# ---------------------------------------------------------------------------

@cuda.jit
def add_sy(sxy, szy, dvydx, dvydz, mu, dt, npt):
    k = idx()
    if k < npt:
        sxy[k] += dt * mu[k] * dvydx[k]
        szy[k] += dt * mu[k] * dvydz[k]


@cuda.jit
def add_sy_c(syx_c, syy_c, syz_c, vy_c, dvydx_c, dvydz_c, dvxdz, dvzdx, dvzdz, nu, mu_c, nu_c, dt):
    k = idx()
    if k < syx_c.size:
        syy_c[k] += 2 * dt * nu[k] * (-vy_c[k] - 0.5 * (dvzdx[k] - dvxdz[k]))
        syx_c[k] += dt * (mu_c[k] + nu_c[k]) * dvydx_c[k]
        syz_c[k] += dt * (mu_c[k] + nu_c[k]) * dvydz_c[k]


@cuda.jit
def add_sxz(sxx, szz, sxz, dvxdx, dvxdz, dvzdx, dvzdz, lam, mu, dt):
    k = idx()
    if k < sxx.size:
        sxx[k] += dt * ((lam[k] + 2 * mu[k]) * dvxdx[k] + lam[k] * dvzdz[k])
        szz[k] += dt * ((lam[k] + 2 * mu[k]) * dvzdz[k] + lam[k] * dvxdx[k])
        sxz[k] += dt * (mu[k] * (dvxdz[k] + dvzdx[k]))


# ---------------------------------------------------------------------------
# seismogram recording
# ---------------------------------------------------------------------------

@cuda.jit
def save_obs(obs, u, rec_id, it, nt, nx, nz):
    ib = cuda.blockIdx.x
    kr = ib * nt + it
    km = rec_id[ib]
    obs[kr] = u[km]


@cuda.jit
def save_obs_c(obs_i, obs_c, u, duzdx, duxdz, rec_id, it, nt, nx, nz):
    ib = cuda.blockIdx.x
    kr = ib * nt + it
    km = rec_id[ib]
    obs_c[kr] = -(duzdx[km] - duxdz[km])
    obs_i[kr] = u[km] - 0.5 * obs_c[kr]


# ---------------------------------------------------------------------------
# kernel accumulation
# ---------------------------------------------------------------------------

@cuda.jit
def interaction_muy(k_mu, dvydx, dvydx_fw, dvydz, dvydz_fw, ndt, nx, nz):
    k = idx()
    if k < nx * nz:
        k_mu[k] += (dvydx[k] * dvydx_fw[k] + dvydz[k] * dvydz_fw[k]) * ndt


# ---------------------------------------------------------------------------
# time-stepping loops (plain Python, called on the solver object)
# ---------------------------------------------------------------------------

def run_forward(self):
    """Run forward simulation for one task (source index = self.taskid)."""
    stream = self.stream
    dim = self.dim
    sh = self.sh
    psv = self.psv
    spin = self.spin

    nx, nz = self.nx, self.nz
    nt = self.nt
    dx, dz, dt = self.dx, self.dz, self.dt

    isrc = -1 if self.config.get('combine_sources') == 'yes' else self.taskid

    npt = self.npt[0]
    out = np.zeros(npt, dtype='float32')
    sfe = int(self.config.get('save_snapshot', 0))
    sae = self.sae

    self.clear_wavefields()

    for it in range(nt):
        isa = -1
        if sae and (it + 1) % sae == 0:
            isa = int(self.nsa - (it + 1) / sae)

        # snapshot save (forward wavefields for adjoint)
        if isa >= 0:
            if sh:
                self.uy.copy_to_host(self.uy_fwd[isa], stream=stream)
            if psv:
                self.ux.copy_to_host(self.ux_fwd[isa], stream=stream)
                self.uz.copy_to_host(self.uz_fwd[isa], stream=stream)

        # SH component
        if sh:
            div_sy[dim](self.dsy, self.sxy, self.szy, dx, dz, nx, nz)
            stf_dsy[self.nsrc, 1](self.dsy, self.stf_y, self.src_id, isrc, it, nt)
            add_vy[dim](self.vy, self.uy, self.dsy, self.rho, self.bound, dt, npt)
            div_vy[dim](self.dvydx, self.dvydz, self.vy, dx, dz, nx, nz)
            add_sy[dim](self.sxy, self.szy, self.dvydx, self.dvydz, self.mu, dt, npt)
            save_obs[self.nrec, 1](self.obs_y, self.uy, self.rec_id, it, nt, nx, nz)

        # P-SV component
        if psv:
            div_sxz[dim](self.dsx, self.dsz, self.sxx, self.szz, self.sxz, dx, dz, nx, nz)

            if spin:
                div_sy[dim](self.dsy_c, self.syx_c, self.syz_c, dx, dz, nx, nz)
                div_sxyz_c[dim](self.dsx, self.dsz, self.dsy_c, self.syy_c, dx, dz, nx, nz)

            stf_dsxz[self.nsrc, 1](self.dsx, self.dsz, self.stf_x, self.stf_z, self.src_id, isrc, it, nt)
            add_vxz[dim](self.vx, self.vz, self.ux, self.uz, self.dsx, self.dsz, self.rho, self.bound, dt)
            div_vxz[dim](self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.vx, self.vz, dx, dz, nx, nz)
            add_sxz[dim](self.sxx, self.szz, self.sxz, self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.lam, self.mu, dt)

            if spin:
                add_vy[dim](self.vy_c, self.uy_c, self.dsy_c, self.j, self.bound, dt, npt)
                div_vy[dim](self.dvydx_c, self.dvydz_c, self.vy_c, dx, dz, nx, nz)
                add_sy_c[dim](self.syx_c, self.syy_c, self.syz_c, self.vy_c,
                               self.dvydx_c, self.dvydz_c,
                               self.dvxdz, self.dvzdx, self.dvzdz,
                               self.nu, self.mu_c, self.nu_c, dt)

            save_obs[self.nrec, 1](self.obs_x, self.ux, self.rec_id, it, nt, nx, nz)
            save_obs[self.nrec, 1](self.obs_z, self.uz, self.rec_id, it, nt, nx, nz)

            if spin:
                save_obs[self.nrec, 1](self.obs_y, self.uy_c, self.rec_id, it, nt, nx, nz)
                div_vxz[dim](self.dvxdx, self.duxdz, self.duzdx, self.dvzdz, self.ux, self.uz, dx, dz, nx, nz)
                save_obs_c[self.nrec, 1](self.obs_yi, self.obs_yc, self.uy_c,
                                          self.duzdx, self.duxdz, self.rec_id, it, nt, nx, nz)

        # trailing snapshot save
        if isa >= 0:
            if sh:
                self.vy.copy_to_host(self.vy_fwd[isa], stream=stream)
            if psv:
                self.vx.copy_to_host(self.vx_fwd[isa], stream=stream)
                self.vz.copy_to_host(self.vz_fwd[isa], stream=stream)

        # periodic field snapshot to disk
        if sfe and it > 0 and it % sfe == 0:
            if sh:
                self.vy.copy_to_host(out, stream=stream)
                stream.synchronize()
                self.export_field(out, 'vy', it)
            if psv:
                self.vx.copy_to_host(out, stream=stream)
                stream.synchronize()
                self.export_field(out, 'vx', it)
                self.vz.copy_to_host(out, stream=stream)
                stream.synchronize()
                self.export_field(out, 'vz', it)
                if spin:
                    self.vy_c.copy_to_host(out, stream=stream)
                    stream.synchronize()
                    self.export_field(out, 'ry', it)

    # write output traces
    if 'output_traces' in self.path:
        tracedir = self.path['output_traces']
        nrec = self.nrec
        i = isrc if isrc >= 0 else 0

        if sh:
            out = np.zeros(nt * nrec, dtype='float32')
            self.obs_y.copy_to_host(out, stream=stream)
            np.save('%s/uy_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

        if psv:
            out = np.zeros(nt * nrec, dtype='float32')
            self.obs_x.copy_to_host(out, stream=stream)
            np.save('%s/ux_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

            out = np.zeros(nt * nrec, dtype='float32')
            self.obs_z.copy_to_host(out, stream=stream)
            np.save('%s/uz_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

            if spin:
                out = np.zeros(nt * nrec, dtype='float32')
                self.obs_y.copy_to_host(out, stream=stream)
                np.save('%s/ry_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

                out = np.zeros(nt * nrec, dtype='float32')
                self.obs_yi.copy_to_host(out, stream=stream)
                np.save('%s/iry_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

                out = np.zeros(nt * nrec, dtype='float32')
                self.obs_yc.copy_to_host(out, stream=stream)
                np.save('%s/cry_%06d.npy' % (tracedir, i), out.reshape([nrec, nt]))

    stream.synchronize()


def run_adjoint(self):
    """Run adjoint simulation (requires setup_adjoint() already called)."""
    stream = self.stream
    dim = self.dim
    sh = self.sh
    psv = self.psv

    nx, nz = self.nx, self.nz
    nt = self.nt
    dx, dz, dt = self.dx, self.dz, self.dt

    npt = self.npt[0]
    sae = self.sae

    self.clear_wavefields()

    for it in range(nt):
        if sh:
            div_sy[dim](self.dsy, self.sxy, self.szy, dx, dz, nx, nz)
            stf_dsy[self.nrec, 1](self.dsy, self.adstf_y, self.rec_id, -1, it, nt)
            add_vy[dim](self.vy, self.uy, self.dsy, self.rho, self.bound, dt, npt)
            div_vy[dim](self.dvydx, self.dvydz, self.vy, dx, dz, nx, nz)
            add_sy[dim](self.sxy, self.szy, self.dvydx, self.dvydz, self.mu, dt, npt)

            if (it + sae) % sae == 0:
                isae = int((it + sae) / sae - 1)
                ndt = sae * dt
                self.dsy.copy_to_device(self.uy_fwd[isae], stream=stream)
                div_vy[dim](self.dvydx, self.dvydz, self.uy, dx, dz, nx, nz)
                div_vy[dim](self.dvydx_fw, self.dvydz_fw, self.dsy, dx, dz, nx, nz)
                interaction_muy[dim](self.k_mu, self.dvydx, self.dvydx_fw,
                                      self.dvydz, self.dvydz_fw, ndt, nx, nz)

        if psv:
            div_sxz[dim](self.dsx, self.dsz, self.sxx, self.szz, self.sxz, dx, dz, nx, nz)
            stf_dsxz[self.nsrc, 1](self.dsx, self.dsz, self.stf_x, self.stf_z, self.src_id, -1, it, nt)
            add_vxz[dim](self.vx, self.vz, self.ux, self.uz, self.dsx, self.dsz, self.rho, self.bound, dt)
            div_vxz[dim](self.dvxdx, self.dvxdz, self.dvzdx, self.dvzdz, self.vx, self.vz, dx, dz, nx, nz)
            add_sxz[dim](self.sxx, self.szz, self.sxz, self.dvxdx, self.dvxdz,
                          self.dvzdx, self.dvzdz, self.lam, self.mu, dt)


def run_kernel(self, adj):
    """
    Run forward + (optionally) adjoint simulations for all sources.

    Parameters
    ----------
    adj : bool
        If True, also run adjoint and accumulate sensitivity kernels.

    Returns
    -------
    misfit : float
        Total waveform misfit.
    k_mu, mu : cuda device arrays  (only when adj=True)
    """
    if not self.mpi:
        print('Computing kernels' if adj else 'Computing misfit')

    stream = self.stream
    nsrc = self.nsrc
    sh = self.sh
    psv = self.psv

    self.setup_adjoint()
    self.clear_kernels()
    misfit = 0.0

    for i in range(nsrc):
        if self.mpi and self.mpi.rank() != i:
            continue

        if not self.mpi:
            print('  task %02d / %02d' % (i + 1, nsrc))

        self.taskid = i
        self.run_forward()

        j = 0 if self.mpi else i

        if sh:
            misfit += self._compute_misfit('y', self.syn_y[j])
            if adj:
                self.run_adjoint()

        if psv:
            misfit += self._compute_misfit('x', self.syn_x[j])
            misfit += self._compute_misfit('z', self.syn_z[j])
            if adj:
                self.run_adjoint()

    if self.mpi:
        mf_sum = np.zeros(1, dtype='float32')
        self.mpi.sum(misfit.astype('float32'), mf_sum)
        misfit = mf_sum[0]

        gradient = np.zeros(self.nx * self.nz, dtype='float32')
        self.k_mu.copy_to_host(gradient, stream=stream)
        stream.synchronize()

        rank = self.mpi.rank()
        fname = self.path['output'] + '/tmp/mu_' + str(rank) + '.npy'
        gradient.tofile(fname)

        self.mpi.sync()
        for i in range(self.nsrc):
            fname = self.path['output'] + '/tmp/mu_' + str(i) + '.npy'
            gradient += np.fromfile(fname, dtype='float32')

        self.k_mu = cuda.to_device(gradient, stream=stream)

    if adj:
        self.smooth(self.k_mu)

    if not self.mpi:
        print('  misfit = %.2f' % misfit)

    if adj:
        return misfit, self.k_mu, self.mu
    return misfit
