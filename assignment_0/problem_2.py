#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:42:40 2024

@author: Rabin Meetarbhan

Note: Requires "problem_1_output.txt" to be present in the same directory
"""

import numpy as np
from matplotlib import pyplot as plt

x, y = np.loadtxt("problem_1_output.txt", unpack=True)

plt.plot(x, y, color="red")

plt.xlim(left=-5, right=5)
plt.xticks(np.linspace(-5, 5, 11))
plt.grid()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("$f(x) = e^{-x^2 / 2}$")

plt.savefig("problem_2_output.pdf")

