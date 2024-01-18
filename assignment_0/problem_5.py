#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:07:08 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

x_max = np.pi
x_min = -np.pi

x_spline = np.linspace(x_min, x_max, 100)
y_n100 = np.sin(x_spline)

# 10 sample interpolation

x1 = np.linspace(x_min, x_max, 11, endpoint=False)[1:]
y1 = np.sin(x1)

y_spline1 = sp.interpolate.CubicSpline(x1, y1)
delta1 = y_spline1(x_spline) - y_n100
MAE1 = np.max(np.abs(delta1))               # Maximum absolute error

# 20 sample interpolation

x2 = np.linspace(x_min, x_max, 21, endpoint=False)[1:]
y2 = np.sin(x2)

y_spline2 = sp.interpolate.CubicSpline(x2, y2)
delta2 = y_spline2(x_spline) - y_n100
MAE2 = np.max(np.abs(delta2))

plt.plot(x_spline, delta1, color="blue", label=f"10 samples\n(MAE = {MAE1:.5f}\nat endpoints)")
plt.plot(x_spline, delta2, color="green", label=f"20 samples\n(MAE = {MAE2:.5f}\nat endpoints)")

plt.xlim(left=x_min, right=x_max)
plt.grid()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Difference Between $y = \sin(x)$ and its Cubic Interpolation\n ($y_{spline} - y$) for 10 and 20 Sample Points")
plt.legend()

plt.savefig("problem_5_output.pdf")

