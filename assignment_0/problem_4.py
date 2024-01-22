#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:32:31 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt

x_max = np.pi
x_min = -np.pi

x = np.linspace(x_min, x_max, 11, endpoint=False)[1:]
y = np.sin(x)

x_spline = np.linspace(x_min, x_max, 100)
y_spline = CubicSpline(x, y)

plt.plot(x, y, ".", color="black", zorder=3)
plt.plot(x_spline, y_spline(x_spline), color="red")

plt.xlim(left=x_min, right=x_max)
plt.grid()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Cubic Interpolation of $y = \sin(x)$")

plt.savefig("problem_4_output.pdf")

