#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:49:38 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gamma
import matplotlib.pyplot as plt
import aux

def zero_crossing_idx(arr):
    """
    Returns the index/indices of an array where a zero is found.
    """
    
    n = len(arr)
    mask = np.nonzero(np.isclose(np.zeros(n), arr, atol=0.01))
    
    return mask

def positive_deriv_idx(arr):
    """
    Returns the index/indices of an array where a positive derivative is found.
    """
    
    grad = np.gradient(arr)
    mask = np.nonzero(grad > 0)
    
    return mask
    
amp = 1

dxdt = lambda t, x : aux.deriv(t, x, 1.0)

n = 100
t = np.linspace(0, 4, n)
dt = t / n

sol = solve_ivp(dxdt, (t[0], t[-1]), [amp, 0], method="Radau", t_eval=t, atol=1E-12)
x = sol.y[0, :]
v = sol.y[1, :]

idx1 = zero_crossing_idx(v)
idx2 = positive_deriv_idx(v)

idx = np.intersect1d(idx1, idx2)
idx_max = idx[-1]

period = 2 * t[idx_max] - t[0]
period_series = np.square(gamma(0.25)) / np.sqrt(np.pi)

rel_err = (period - period_series) / period_series

print(f"Period = {period} (integration)")
print(f"Period = {period_series} (exact)")
print(f"Relative error = {rel_err}")
    
fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
ax.plot(t[:idx_max + 1], x[:idx_max + 1], color="red", label="Displacement")
ax.plot(t[:idx_max + 1], v[:idx_max + 1], color="blue", label="Velocity")
ax.set_xlabel("Time")
ax.set_ylabel("Displacement")
ax.set_title("Normalized Motion of a Quadric Oscillator")
ax.grid()
ax.legend(frameon=False)
plt.savefig("problem_2_output.pdf", bbox_inches="tight")