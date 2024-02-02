#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:29:09 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt
import aux

data = aux.load_data_stdin()

k_max = 7

def basis_fn(x, degree):
    return np.cos(2 * np.pi * x * np.arange(0, degree + 1, 1))

x = data[:,0]
y = data[:,1]

a = aux.basis_vander(x, basis_fn, k_max)
c = np.dot(np.linalg.pinv(a), y)

cd_number = aux.cd_number(a)

print(f"Condition number for matrix A: {cd_number}")

x_fit = np.linspace(0, 1, 100)
y_fit = aux.basis_val(x_fit, basis_fn, c)

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(x, y, ".", markersize=1, color="black")
plt.plot(x_fit, y_fit, color="red", label=f"Order {k_max} approximation")
plt.xlim(0, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Plot of y versus x")
plt.grid()
plt.legend()
plt.savefig("problem_4_2_output.pdf")