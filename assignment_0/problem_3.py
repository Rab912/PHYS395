#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:59:44 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from matplotlib import pyplot as plt

n = 10                      # Number of Legendre polynomials

x = np.linspace(-1, 1, 100)
y = np.polynomial.legendre.legvander(x, n)

plt.plot(x, y)

plt.xlim(left=-1, right=1)
plt.grid()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Plot of first 10 legendre Polynomials")

plt.savefig("problem_3_output.pdf")

