#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:25:22 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + 10 * np.square(x))

# Finite-difference derivative approximation for Chebyshev approximation
def f_deriv_fd(func, x):
    y = func(x)
    yp = np.empty(np.size(x))
    
    # First-order approximation at the endpoints
    
    yp[0] = (y[1] - y[0]) / (x[1] - x[0])
    yp[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    
    # Second-order approximation within the domain, excluding endpoints
    
    for i in range(1, np.size(x) - 1):
        yp[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    
    return yp

# Used for plotting purposes only - not used for the Chebyshev approximation
def f_deriv(x):
    return -20 * x / np.square(1 + 10 * np.square(x))

# 100 samples (and terms)
n = 100

grid = np.cos((np.arange(0, n) + 0.5) * np.pi / n)

f_deriv_data = f_deriv_fd(f, grid)

vm_mat = np.polynomial.chebyshev.chebvander(grid, n - 1)
f_deriv_cheb_coeff = np.linalg.solve(vm_mat, f_deriv_data)

f_deriv_cheb = np.polynomial.chebyshev.chebval(grid, f_deriv_cheb_coeff)

fig, axs = plt.subplots(2)

axs[0].plot(grid, f_deriv(grid), color="red")
axs[1].plot(grid, f_deriv_cheb, color="blue")

axs[0].set_xlim(left=-1, right=1)
axs[1].set_xlim(left=-1, right=1)
axs[0].set_ylabel("$y$")
axs[1].set_ylabel("$y$")
axs[1].set_xlabel("$x$")
fig.suptitle("Chebyshev Approximation of $f'(x) = -20x/(1 + 10x^2)^2$" + f" of Degree {n}")
axs[0].text(0.8, 0.8, "$f'(x)$")
axs[1].text(0.3, 0.8, "Approximation of $f'(x)$\n(with finite-difference)")
axs[0].grid()
axs[1].grid()

plt.savefig("problem_4_part_2_output.pdf")

