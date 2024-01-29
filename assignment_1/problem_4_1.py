#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:23:29 2024

@author: Rabin Meetarbhan

Note: This script does not output a file. Instead, it outputs to the standard output stream (console).
"""

import numpy as np
from numpy.polynomial import chebyshev as ch
import aux

f = lambda x : 1 / (1 + 10 * np.square(x))

grid_10 = np.cos((np.arange(0, 10) + 0.5) * np.pi / 10)
grid_100 = np.cos((np.arange(0, 100) + 0.5) * np.pi / 100)

coeff_10 = aux.cheb_coefficients(grid_10, f(grid_10))
coeff_100 = aux.cheb_coefficients(grid_100, f(grid_100))

print("Order 10 approximation coefficients:\n", coeff_10)
print("\nOrder 100 approximation coefficients:\n", coeff_100)

# --- Optional plot generation to visualize approximations

gen_plot = False

if (gen_plot):
    import matplotlib.pyplot as plt
    
    # Fine grid for plotting
    x = np.linspace(-1, 1, 1000)
    
    f_actual = f(x)
    f_cheb_10 = ch.chebval(x, coeff_10)
    f_cheb_100 = ch.chebval(x, coeff_100)
    
    fig, axs = plt.subplots(2, sharex=True, constrained_layout=True)
    
    axs[0].plot(x, f_actual, color="black", alpha=0.3)
    axs[0].plot(x, f_cheb_10, color="blue")
    
    axs[1].plot(x, f_actual, color="black", alpha=0.3)
    axs[1].plot(x, f_cheb_100, color="red")
    
    axs[0].set_xlim(left=-1.1, right=1.1)
    axs[0].set_ylim(bottom=-0.1, top=1.1)
    axs[1].set_ylim(bottom=-0.1, top=1.1)
    
    axs[0].tick_params(axis='x', which='both', bottom=False)
    axs[1].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[1].set_ylabel("y")
    axs[0].set_title("Chebyshev Approximation of Order 10 for $f(x)$")
    axs[1].set_title("Chebyshev Approximation of Order 100 for $f(x)$")
    axs[0].grid()
    axs[1].grid()

    plt.show()

