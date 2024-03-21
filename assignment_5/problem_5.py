#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:35:11 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def chebyshev(x, y, degree):
    """
    A set of coupled first-order ODES which represent the Chebyshev differential equation.
    """
    
    y1, y2 = y
    
    return [y2, (x * y2 - y1 * degree ** 2) / (1 - x ** 2)]

def pad(arr, size, value=0):
    """
    Adds padding to the end of an array.

    All half-order polynomials slam instantaneously to y = 1 at x = 1. The integrator does not like this and therefore excludes the last (x = 1) element from the solution.
    """
    
    n = len(arr)
    
    if size > n:
        padding = value * np.ones(size - n)
        
        return np.append(arr, padding)
    
    return arr

maxdeg = 7
n = 100
y = np.empty((0, n))
y_fr = np.empty((0, n))
x = np.linspace(0, 1, n)

# Integer-order polynomials
for i in range(maxdeg):
    ic = [1, 0] if i % 2 == 0 else [0, 1]
    sol = solve_ivp(chebyshev, (x[0], x[-1] + 0.1), ic, t_eval=x, args=(i,))
    y = np.vstack([y, sol.y[0, :]])
    y[i] /= y[i, -1]

# Half-order polynomials
# These do not have a normalization because they are undefined at x = 1
# This could mean they actually do not exist
# The curves were still generated, so will show them
for i in range(maxdeg):
    ic = [1, 0] if i % 2 == 0 else [0, 1]
    sol = solve_ivp(chebyshev, (x[0], x[-2]), ic, t_eval=x[:-2], args=(i + 0.5,))
    y_fr = np.vstack([y_fr, pad(sol.y[0, :], 100, 1)])
 
fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=90)
label = [r"$\nu$ = " + f"{i}" for i in range(maxdeg)]
ax[0].plot(x, y.T, label=label)
ax[0].set_xlabel("$x$")
ax[0].set_ylabel("$y$")
ax[0].set_title(f"First {maxdeg} Chebyshev Polynomials on $x \in$ [0, 1]")
ax[0].set_xlim(0, 1)
ax[0].grid()
ax[0].legend(frameon=False, loc="lower left")

label = [r"$\nu$ = " + f"{i + 0.5}" for i in range(maxdeg)]
ax[1].plot(x[:-2], y_fr[:, :-2].T, label=label)
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$y$")
ax[1].set_title(f"First {maxdeg} Half-Order Chebyshev Polynomials on\n$x \in$ [0, 1] (Not Normalized)")
ax[1].set_xlim(0, 1)
ax[1].grid()
ax[1].legend(frameon=False, loc="lower left")

plt.tight_layout()
plt.savefig("problem_5_output.pdf", bbox_inches="tight")