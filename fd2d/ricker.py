"""Ricker (Mexican-hat) source-time function."""

import numpy as np


def ricker(t, f0, t0, ang, amp):
    """
    Evaluate a Ricker wavelet at time *t*.

    Parameters
    ----------
    t   : float – current time
    f0  : float – dominant frequency [Hz]
    t0  : float – time delay [s]
    ang : float – incidence angle [degrees]
    amp : float – amplitude scaling

    Returns
    -------
    (stf_x, stf_y, stf_z) : tuple of floats
        Force components in x, y (SH), and z directions.
    """
    a = (np.pi * f0) ** 2
    stf = -amp * (1.0 - 2.0 * a * (t - t0) ** 2) * np.exp(-a * (t - t0) ** 2)
    ang_rad = np.deg2rad(ang)
    return stf * np.sin(ang_rad), stf, -stf * np.cos(ang_rad)
