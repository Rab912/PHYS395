#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:26:43 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + 10 * np.square(x))

# Approximation with 10 and 100 coefficients

i = 1

for n in [10, 100]:
    grid = np.linspace(-1, 1, n)

    f_data = f(grid)

    # The number of terms in the approximation must match the number of sample
    # points used. Otherwise, the approximating polynomial becomes over/under-
    # constrained (vm_mat becomes non-square and hence will have no inverse).
    vm_mat = np.polynomial.chebyshev.chebvander(grid, n - 1)
    f_cheb_coeff = np.linalg.solve(vm_mat, f_data)

    f_cheb = np.polynomial.chebyshev.chebval(grid, f_cheb_coeff)

    fig, axs = plt.subplots(2)
    
    print(f"\nOrder {n} Chebyshev approximation coefficients:")
    
    for j in range(n):
        print(f"c_{j} =", f_cheb_coeff[j])
    
    axs[0].plot(grid, f_data, color="red")
    axs[1].plot(grid, f_cheb, color="blue")

    axs[0].set_xlim(left=-1, right=1)
    axs[1].set_xlim(left=-1, right=1)
    axs[0].set_ylim(top=1.1)
    axs[1].set_ylim(top=1.1)
    axs[0].set_ylabel("$y$")
    axs[1].set_ylabel("$y$")
    axs[1].set_xlabel("$x$")
    fig.suptitle("Chebyshev Approximation of $f(x) = 1/(1 + 10x^2)$" + f" of Degree {n}")
    axs[0].text(0.8, 0.8, "$f(x)$")
    axs[1].text(0.3, 0.8, "Approximation of $f(x)$")
    axs[0].grid()
    axs[1].grid()
    
    plt.savefig(f"problem_1_output{i}.pdf")
    
    i = i + 1

