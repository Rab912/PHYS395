# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 01:18:55 2024

@author: Rabin Meetarbhan
"""

import sys
import numpy as np

"""
Loads in data through the standard input.
"""
def load_data_stdin():
    data = np.loadtxt(sys.argv[1])

    length, columns = data.shape

    if columns < 2:
        raise ValueError("Supplied data file must contain at least two columns")
        
    return data

"""
Generates a Vandermonde matrix for a given set of basis functions up to some degree.

The given basis function must have the parameter list (<x data>, <highest term>).
"""
def basis_vander(x, basis, degree):
    a = np.empty((np.size(x), degree + 1))
    
    for i in range(np.size(x)):
        a[i] = basis(x[i], degree)
        
    return a

"""
Evaluates a basis function series over a set of points.

The given basis function must have the parameter list (<x data>, <highest term>).
"""
def basis_val(x, basis, c):
    return np.array([np.dot(c, basis(x[i], np.size(c) - 1)) for i in range(np.size(x))])

"""
Calculates the condition number for a matrix.
"""
def cd_number(mat):
    s = np.linalg.svd(mat, compute_uv=False)
    
    return s[0] / s[-1]

"""
Returns the chi-squared value for a fit to data. Currently uses weights equal to 1.
"""
def chi_squared(y, y_model):
    return np.sum(np.square(y - y_model))

"""
Explicitly calculates the inverse of the SVD of the given matrix (in case this is required).
"""
def svd_inv(a):
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    
    return np.dot(vh.T, np.dot(np.diag(np.reciprocal(s, where=s!=0)), u.T))

