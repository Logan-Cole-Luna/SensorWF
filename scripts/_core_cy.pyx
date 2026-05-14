# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
_core_cy.pyx -- Compiled inner loops for SensorWF pipeline_core.

Build
-----
    python setup_cy.py build_ext --inplace

Exposes
-------
rolling_skew_kurt(x2d, window, min_sk=-1, min_ku=-1)
    O(n) rolling skewness and excess kurtosis for a (n_samples, n_channels)
    float64 array.  Uses a sliding-window running-sum over x, x², x³, x⁴ so
    each sample is visited exactly twice (once added, once removed).

    Replaces the pandas rolling().skew() / rolling().kurt() path in
    compute_extended_features() — typically 4–8× faster on arrays with many
    channels or long sessions.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


def rolling_skew_kurt(
    cnp.ndarray[DTYPE_t, ndim=2] x2d not None,
    int window,
    int min_sk = -1,
    int min_ku = -1,
):
    """
    O(n) rolling population skewness and excess kurtosis.

    Parameters
    ----------
    x2d    : (n_samples, n_channels) float64, C-contiguous
    window : rolling window length in samples
    min_sk : minimum samples before skewness is emitted (default: window // 3)
    min_ku : minimum samples before kurtosis is emitted (default: window // 4)

    Returns
    -------
    skew_arr, kurt_arr : both (n_samples, n_channels) float64, zero-padded
    """
    cdef int n  = x2d.shape[0]
    cdef int nc = x2d.shape[1]

    if min_sk < 0:
        min_sk = max(3, window // 3)
    if min_ku < 0:
        min_ku = max(4, window // 4)

    cdef cnp.ndarray[DTYPE_t, ndim=2] skew_arr = np.zeros((n, nc), dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=2] kurt_arr = np.zeros((n, nc), dtype=DTYPE)

    cdef double[:, :] x  = x2d
    cdef double[:, :] sk = skew_arr
    cdef double[:, :] ku = kurt_arr

    cdef int i, ci, nw
    cdef double xi, xo, s1, s2, s3, s4
    cdef double mu, var, m3, m4

    for ci in range(nc):
        s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
        for i in range(n):
            # Add incoming sample
            xi  = x[i, ci]
            s1 += xi
            s2 += xi * xi
            s3 += xi * xi * xi
            s4 += xi * xi * xi * xi

            # Evict outgoing sample once window is full
            if i >= window:
                xo  = x[i - window, ci]
                s1 -= xo
                s2 -= xo * xo
                s3 -= xo * xo * xo
                s4 -= xo * xo * xo * xo

            nw  = i + 1 if i < window else window
            mu  = s1 / nw
            var = s2 / nw - mu * mu   # population variance

            if nw >= min_sk and var > 1e-14:
                # Population skewness = E[(x-µ)³] / σ³
                #   E[(x-µ)³] = s3/n - 3µ(s2/n) + 2µ³
                m3 = s3 / nw - 3.0 * mu * (s2 / nw) + 2.0 * mu * mu * mu
                sk[i, ci] = m3 / (var * sqrt(var))

            if nw >= min_ku and var > 1e-14:
                # Population excess kurtosis = E[(x-µ)⁴] / σ⁴ - 3
                #   E[(x-µ)⁴] = s4/n - 4µ(s3/n) + 6µ²(s2/n) - 3µ⁴
                m4 = (s4 / nw
                      - 4.0 * mu * (s3 / nw)
                      + 6.0 * mu * mu * (s2 / nw)
                      - 3.0 * mu * mu * mu * mu)
                ku[i, ci] = m4 / (var * var) - 3.0

    return skew_arr, kurt_arr
