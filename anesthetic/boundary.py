"""Boundary correction utilities."""

import numpy as np
from scipy.special import erf


def cut_and_normalise_gaussian(x, p, bw, xmin=None, xmax=None):
    """Cut and normalise boundary correction for a Gaussian kernel.

    Parameters
    ----------
    x : array-like
        locations for normalisation correction

    p : array-like
        probability densities for normalisation correction

    bw : float
        bandwidth of KDE

    xmin, xmax : float
        lower/upper prior bound
        optional, default None

    Returns
    -------
    p : np.array
        corrected probabilities

    """
    def Phi(z):
        return 0.5*(1 + erf(z/np.sqrt(2)))
    a = (xmin - x)/bw if xmin is not None else -np.inf
    b = (xmax - x)/bw if xmax is not None else +np.inf
    correction = Phi(b) - Phi(a)
    p /= correction

    if xmin is not None:
        p[x < xmin] = 0
    if xmax is not None:
        p[x > xmax] = 0
    return p
