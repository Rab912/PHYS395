# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 01:29:12 2024

@author: Rabin Meetarbhan

Note: This file contains function implementations that are used multiple times in the assignment.
It is not intended to be executed on its own.
"""

import numpy as np
from numpy.polynomial import chebyshev as ch

"""
Calculates the coefficients in the Chebyshev approximation of a sampled function on a given grid.
The order of the approximation is equal to one less than the number of data points.
"""
def cheb_coefficients(x, y):    
    vm_mat = ch.chebvander(x, np.size(x) - 1)
    
    coeff = np.linalg.solve(vm_mat, y)
    
    return coeff

"""
Calculates the error function in a given Chebyshev approximation of a sampled function.
Ignores sign.
"""
def cheb_error(x, y, coeff):
    delta = y - ch.chebval(x, coeff)
    
    return np.abs(delta)

