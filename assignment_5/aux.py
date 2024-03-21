# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:17:32 2024

@author: rab9g
"""

import numpy as np
from scipy.special import gamma

def deriv(t, x, lmb):
    """
    A set of coupled first-order ODEs which represent the second-order equation of motion.

    Derived by applying F = ma with the negative gradient of the potential as the force.
    """
    
    x1, x2 = x
    
    return [x2, -lmb * x1 ** 3]

def jecf(t, tol=1E-12, ret_nterms=False):
    """
    Returns the Jacobi elliptical cosine function up to a specified precision, and optionally the number of terms used in the series.
    """
    
    c = np.sqrt(2) * 4 * np.pi
    p = np.square(gamma(0.25)) / np.sqrt(np.pi)
    
    n = 1
    ssumprev = 0
    ssum = c * np.cos((n - 0.5) * 4 * np.pi * t / p) / (p * np.cosh((n - 0.5) * np.pi))
    
    while np.mean(abs(ssum - ssumprev)) > tol:
        n += 1
        ssumprev = ssum
        ssum += c * np.cos((n - 0.5) * 4 * np.pi * t / p) / (p * np.cosh((n - 0.5) * np.pi))

    if not ret_nterms:
        return ssum

    else:
        return ssum, n