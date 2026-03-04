"""Waveform (L2) misfit and adjoint source-time function."""

import math
import numpy as np
from numba import cuda


@cuda.jit
def _diff(syn, obs, adstf, misfit, nt, dt):
    it = cuda.blockIdx.x
    ir = cuda.threadIdx.x
    kt = ir * nt + it
    akt = ir * nt + nt - it - 1

    t_end = (nt - 1) * dt
    taper_width = t_end / 10
    t_min = taper_width
    t_max = t_end - taper_width
    t = it * dt

    if t <= t_min:
        taper = 0.5 + 0.5 * math.cos(math.pi * (t_min - t) / taper_width)
    elif t >= t_max:
        taper = 0.5 + 0.5 * math.cos(math.pi * (t_max - t) / taper_width)
    else:
        taper = 1.0

    misfit[kt] = (syn[kt] - obs[kt]) * taper
    adstf[akt] = misfit[kt] * taper * 2


def waveform(syn, obs, adstf, nt, dt, nrec, stream):
    """
    Compute L2 waveform misfit and populate the adjoint source *adstf*.

    Parameters
    ----------
    syn, obs, adstf : CUDA device arrays  (flat, length nrec * nt)
    nt, dt          : int, float  – time samples and interval
    nrec            : int         – number of receivers
    stream          : CUDA stream

    Returns
    -------
    float – scalar misfit value
    """
    misfit = np.zeros(nt * nrec, dtype='float32')
    d_misfit = cuda.to_device(misfit, stream=stream)
    _diff[nt, nrec](syn, obs, adstf, d_misfit, nt, dt)
    d_misfit.copy_to_host(misfit, stream=stream)
    stream.synchronize()
    return float(np.linalg.norm(misfit))
