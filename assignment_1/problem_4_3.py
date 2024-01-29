#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:29:51 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from numpy.polynomial import chebyshev as ch
import matplotlib.pyplot as plt
import aux

f = lambda x : 1 / (1 + 10 * np.square(x))
f_deriv = lambda x : -20 * x / np.square(1 + 10 * np.square(x))

grid_10 = np.cos((np.arange(0, 10) + 0.5) * np.pi / 10)
grid_100 = np.cos((np.arange(0, 100) + 0.5) * np.pi / 100)

coeff_10 = aux.cheb_coefficients(grid_10, f(grid_10))
coeff_100 = aux.cheb_coefficients(grid_100, f(grid_100))

coeff_deriv_10 = aux.cheb_coefficients(grid_10, f_deriv(grid_10))
coeff_deriv_100 = aux.cheb_coefficients(grid_100, f_deriv(grid_100))

# --- Plotting

# Fine grid for plotting
x = np.linspace(-1, 1, 1000)

error_10 = aux.cheb_error(x, f(x), coeff_10)
error_100 = aux.cheb_error(x, f(x), coeff_100)

error_deriv_10 = aux.cheb_error(x, f_deriv(x), coeff_deriv_10)
error_deriv_100 = aux.cheb_error(x, f_deriv(x), coeff_deriv_100)

print(f"f(x) order 10 maximum error: y = {np.max(error_10)} at x = {x[np.argmax(error_10)]}")
print(f"f(x) order 100 maximum error: y = {np.max(error_100)} at x = {x[np.argmax(error_100)]}")

print(f"f'(x) order 10 maximum error: y = {np.max(error_deriv_10)} at x = {x[np.argmax(error_deriv_10)]}")
print(f"f'(x) order 100 maximum error: y = {np.max(error_deriv_100)} at x = {x[np.argmax(error_deriv_100)]}")

fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)

axs[0].plot(x, error_10, color="blue", label="Order 10")
axs[0].plot(x, error_100, color="red", label="Order 100")

axs[1].plot(x, error_deriv_10, color="blue", label="Order 10")
axs[1].plot(x, error_deriv_100, color="red", label="Order 100")

axs[0].set_xlim(left=-1.1, right=1.1)
axs[0].set_ylim(bottom=-0.005, top=0.1)
axs[1].set_ylim(bottom=-0.05, top=0.6)

axs[0].tick_params(axis='x', which='both', bottom=False)
axs[1].set_xlabel("x")
axs[0].set_ylabel("y")
axs[1].set_ylabel("y")
axs[0].set_title("Error in Chebyshev Approximations of $f(x)$")
axs[1].set_title("Error in Chebyshev Approximations of $f'(x)$")
axs[0].grid()
axs[1].grid()

axs[0].legend()
axs[1].legend()
plt.savefig("problem_4_3_output.pdf")

