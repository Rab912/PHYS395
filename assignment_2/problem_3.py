#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:10:35 2024

@author: Rabin Meetarbhan

Note: Correct console output is given in the corresponding output text file.
"""

import numpy as np
import aux

data = aux.load_data_stdin()

k_max = 3

def basis_fn(x, degree):
    return np.cos(2 * np.pi * x * np.arange(0, degree + 1, 1))

x = data[:,0]
y = data[:,1]

a = aux.basis_vander(x, basis_fn, k_max)
c = np.dot(np.linalg.pinv(a), y)

chisq = aux.chi_squared(y, aux.basis_val(x, basis_fn, c))

print(f"chi-squared: {chisq}, dof: {np.size(x) - k_max}")
print(f"reduced chi-squared: {chisq / (np.size(x) - k_max)}")

