# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:54:40 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.linalg import solve

def hpotential(x):
    """
    Defines a harmonic potential function.
    """
    
    return 0.5 * x ** 2

def qpotential(x):
    """
    Defines a quartic potential function.
    """
    
    return 0.25 * x ** 4

def stationary(x, state, potential, energy):
    """
    The differential equation satisfied by stationary-state wavefunctions.
    """
    
    p1, p2 = state
    
    return (p2, 2 * (potential(x) - energy) * p1)

def norm(u, w, x):
    """
    Returns the signed norm of an eigenfunction with respect to a weight.
    
    It is signed to make the sign convention of the function consistent.
    """
    return np.sqrt(np.sum(w * u * u)) * np.sign(np.sum(w * (1 + x) * u))

def rq_step(u, mat, w, x):
    """
    Calculates the Rayleigh quotient for an iteration.
    """
    
    return np.sum(w * u * (mat @ u)) / norm(u, w, x)

def rq_iter(mat, eigval0, eigvec0, x, w=None, atol=1E-12):    
    n = len(eigvec0)
    
    if w is None:
        w = np.eye(n)
    
    eigvec = eigvec0 / norm(eigvec0, w, x)
    
    eigvec = solve(mat - eigval0 * np.eye(n), eigvec)
    eigvec = eigvec / norm(eigvec, w, x)
    
    eigval = rq_step(eigvec, mat, w, x)
    eigval_prev = eigval + 2 * atol # to guarantee entry into the loop below
    
    i = 0
    while np.abs(eigval - eigval_prev) > atol:
        eigval_prev = eigval
        
        eigvec = eigvec / norm(eigvec, w, x)
        
        eigvec = solve(mat - eigval * np.eye(n), eigvec)
        eigvec = eigvec / norm(eigvec, w, x)
        
        eigval = rq_step(eigvec, mat, w, x)
        
        i += 1
    
    # = 1 if i % 2 == 0 else -1
    
    return eigval, eigvec

# === UNUSED ===

# 8'th order Butcher tableau (in quad precision)
a4 = np.array([
	 0.869637112843634643432659873054998518E-1,
	-0.266041800849987933133851304769531093E-1,
	 0.126274626894047245150568805746180936E-1,
	-0.355514968579568315691098184956958860E-2,
	 0.188118117499868071650685545087171160E0,
	 0.163036288715636535656734012694500148E0,
	-0.278804286024708952241511064189974107E-1,
	 0.673550059453815551539866908570375889E-2,
	 0.167191921974188773171133305525295945E0,
	 0.353953006033743966537619131807997707E0,
	 0.163036288715636535656734012694500148E0,
	-0.141906949311411429641535704761714564E-1,
	 0.177482572254522611843442956460569292E0,
	 0.313445114741868346798411144814382203E0,
	 0.352676757516271864626853155865953406E0,
	 0.869637112843634643432659873054998518E-1
]).reshape([4,4])

b4 = np.array([
	 0.173927422568726928686531974610999704E0,
	 0.326072577431273071313468025389000296E0,
	 0.326072577431273071313468025389000296E0,
	 0.173927422568726928686531974610999704E0
])

def gaussleg8(func, state, dt, args=None):
    """
    Computes an 8'th order Gauss-Legendre step.
    
    Taken from the course GitHub repository.
    """
    
    n = len(state)
    g = np.zeros([4, n])
    
    for k in range(16):
        g = np.matmul(a4, g)
        
        for i in range(4):
            g[i] = func(state + g[i] * dt, *args)
            
    return tuple(state + np.dot(b4, g) * dt)

def evolve_system(func, state0, t_span, dt, args=None):
    """
    Function to solve an initial-value problem using the 8'th order Gauss-Legendre
    integrator step.
    """
    n = int((t_span[-1] - t_span[0]) / dt)
    state_arr = np.empty((len(state0), n))
    
    state = np.copy(state0)
    
    for i in range(n):
        state_arr[:, i] = state
        state = gaussleg8(func, state, dt, *args)
        
    return state_arr