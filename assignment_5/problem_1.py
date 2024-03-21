#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:40:23 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import aux
    
dxdt = lambda t, x : aux.deriv(t, x, 1)

n = 100
t = np.linspace(0, 10, n)
x = np.empty((0, n))

amps = [0.5, 1, 2]

for amp in amps:
    sol = solve_ivp(dxdt, (t[0], t[-1]), [amp, 0], method="Radau", t_eval=t, atol=1E-12)
    x = np.vstack([x, sol.y[0, :]])

tseries = np.linspace(0, 10, 1000)
xseries, nterms = aux.jecf(tseries, ret_nterms=True)

print(f"lambda = 1: {nterms} terms to achieve precision of 1E-12")

fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
ax.plot(t, x.T / amps, label=["Amplitude = 0.5", "Amplitude = 1", "Amplitude = 2"])
# This looks exactly like the lambda = 1 numerical solution. Uncomment to see
#ax.plot(tseries, xseries, color="red", label="$\lambda = 1$ (JECF)")
ax.set_xlabel("Time")
ax.set_ylabel("Displacement")
ax.set_title("Normalized Displacement versus Time of a\nQuadric Oscillator")
ax.grid()
ax.legend(frameon=False)
plt.savefig("problem_1_output.pdf", bbox_inches="tight")