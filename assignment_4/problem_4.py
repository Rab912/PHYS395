# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:34:54 2024

@author: Rabin Meetarbhan
"""

import sys
import numpy as np
from numpy.linalg import inv

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
Template fitting function.
"""
def _tfunc(x, basis_fn, c):
    coeffs = c[:-1]
    bx = basis_fn(x, len(coeffs) - 1)
    lc = np.dot(coeffs, bx)
    
    return np.exp(lc) + c[-1]

"""
First derivative with respect to a coefficient of the template fitting function.
"""
def _tdfunc(x, basis_fn, c, term):
    coeffs = c[:-1]
    m = len(coeffs)
    
    # If differentiating wrt added constant
    if term == m:
        return np.ones(np.size(x))
    
    bx = basis_fn(x, m - 1)
    lc = np.dot(coeffs, bx)
    
    mask = np.zeros(m)
    mask[term] = 1.0
        
    bx0 = np.dot(mask, bx)
    
    return bx0 * np.exp(lc)

"""
Calculates the alpha matrix, which is equal to half the Hessian.
"""
def _alpha(x, basis_fn, c, w):
    n = len(c)
    a = np.empty((n, n))
    
    for i in range(n):
        for j in range(n):
            a[i, j] = np.dot(w, _tdfunc(x, basis_fn, c, i) * _tdfunc(x, basis_fn, c, j))
    
    return a

"""
Calculates the beta vector, which is equal to negative one half the chi-squared gradient.
"""
def _beta(x, basis_fn, y, c, w):
    res = y - _tfunc(x, basis_fn, c)
    m = len(c)
    b = np.empty(m)
    
    for i in range(m):
        b[i] = np.dot(w, res * _tdfunc(x, basis_fn, c, i))
    
    return b

"""
Calculates the chi-squared statistic.
"""
def _chisq(x, basis_fn, y, c, w):
    res = y - _tfunc(x, basis_fn, c)
    
    return np.dot(w, res * res)

"""
Calculates a non-linear least-squares fit of the given exponential-type function using the Levenberg-Marquardt algorithm.
"""
def lmexpfit(basis_fn, x, y, c0, errs):
    n = len(c0)
    w = 1.0 / (errs * errs)
    c = np.copy(c0).astype(np.float64)
    
    chisqprev = _chisq(x, basis_fn, y, c, w)
    chisq = 0
    dchisq = chisq - chisqprev
    
    lmb = 0.001
    
    i = 0
    while abs(dchisq) > 0.001 and i < 1000:    
        alphapr = _alpha(x, basis_fn, c, w)
        
        for j in range(n):
            alphapr[j, j] *= 1.0 + lmb
        
        beta = _beta(x, basis_fn, y, c, w)
        
        dc = np.dot(inv(alphapr), beta)
        
        chisq = _chisq(x, basis_fn, y, c + dc, w)
        
        if chisq > chisqprev:
            lmb *= 10
        
        else:
            lmb *= 0.1
            c += dc
        
        dchisq = chisq - chisqprev
        chisqprev = chisq
        
        i += 1
        
    alpha = _alpha(x, basis_fn, c, w)
    cov = inv(alpha)
    chisq = _chisq(x, basis_fn, y, c, w)
    
    return c, cov, chisq
    
# === UNUSED: Implementing the general LMA would take too long, and would be inaccurate ===

"""
Calculates the Jacobian of a function for a given set of points using finite differences.
"""
def jacobian(func, x, p):
    m = np.size(x)
    n = np.size(p)
    
    jac = np.empty((m, n))
    
    # Assuming function precision is equal to machine precision here
    delta = np.power(np.finfo(float).eps, 1.0 / 3) * p
    h = np.zeros(n)
    
    for i in range(m):
        for j in range(n):
            # Vary one parameter at a time
            h[j] = delta[j]
        
            fl = func(x, p - h)
            fr = func(x, p + h)
        
            df = fr - fl
        
            jac[i, j] = 0.5 * df[i] / h[j]
            
            h[j] = 0.0
    
    return jac

"""
Incrementally changes the Jacobian for a change in the function parameters using Broyden's method.

This is faster than recalculating it from scratch at a new point.

See https://en.wikipedia.org/wiki/Broyden%27s_method.
"""
def bmupdate(jac, df, dp):
    ddf = np.outer(df - jac @ dp, dp)
    dpn = np.dot(dp, dp)
    
    return jac + ddf / dpn