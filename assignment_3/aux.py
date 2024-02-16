# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:56:21 2024

@author: Rabin Meetarbhan
"""

import numpy as np

"""
Generates a histogram of the given data, with equal bin widths.
"""
def density_hist(data, bin_count):
    pdf, bins = np.histogram(data, bins=bin_count, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return bin_centers, pdf

"""
Approximates all sample L-moments up to a specified order for a given array using Legendre polynomials.

Adapted from the class GitHub repository.
"""
def lmoments(x, order, prec=5):
    x_len = np.size(x)
    x_sorted = np.sort(x)
    p = np.polynomial.legendre.legvander(np.linspace(-1.0, 1.0, x_len), prec)
    
    lm = np.empty(order)
    
    for i in range(order):
        lm[i] = np.sum(x_sorted * p[:,i])
        
    return lm / x_len