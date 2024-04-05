#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:48:35 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from numpy.linalg import solve, eig
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import warnings
import aux

warnings.filterwarnings('ignore')

# Increasing the number of levels computed increases resolution of the solutions.
n = 100
cf = 1

dt = np.pi / n
t = np.linspace(np.pi - 0.5 * dt, 0.5 * dt, n)

b = np.empty([n, n])
ddb = np.empty([n, n])

# === Generating Laplacian matrix representation wrt Chebyshev basis ===

for k in range(n):
    b[k, :] = np.cos(k * t)
    ddb[k, :] = -(k * np.sin(t) * np.cos(k * t) +
                  2 * np.cos(t) * np.sin(k * t)) * np.sin(t) ** 3 * (k / cf ** 2)
    
lapl = solve(b, ddb).T

# === Finding energy eigenvalues from Hamiltonian (quartic potential) ===

x = cf / np.tan(t)

hamilt = -0.5 * lapl + np.diag(aux.qpotential(x))

energy, eigf = eig(hamilt)
idx = np.argsort(energy)

# === Finding and normalizing stationary states (eigenstates of Hamiltonian) ===

w = cf / np.sin(t) ** 2

nlevels = 10

energies = np.empty(nlevels)
wave_funcs = np.empty((nlevels, n))
probabilities = np.empty((nlevels, n))

for level in range(nlevels):
    # Initial guess
    e0 = energy[idx[level]]
    wf0 = eigf[idx[level]]
    
    e, wf = aux.rq_iter(hamilt, e0, wf0, x, w=w)
    
    energies[level] = e
    
    nf = simpson(wf * wf, x)
    wave_funcs[level, :] = wf / np.sqrt(nf)
    probabilities[level, :] = (wf ** 2) / nf

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 7), dpi=90)
plt.subplots_adjust(wspace=0.05)

for level in range(nlevels):
    # Not plotted to scale
    ax[0].plot(x, level + 0.5 * wave_funcs[level, :], color="black")
    ax[1].plot(x, level + 1.5 * probabilities[level, :], color="red")
    ax[0].text(3.5, level + 0.35, r"$E$ = " + f"{energies[level]:.5f}", fontsize=8)

ax[0].set_yticks(np.arange(0, 10, 1))
ax[0].set_xlim(-6, 6)
ax[0].set_xlabel(r"$x$")
ax[0].set_title("Probability Amplitude")

ax[1].set_xlim(-6, 6)
ax[1].set_xlabel(r"$x$")
ax[1].set_title("Probability Density")

ax[0].set_ylabel("Energy Number")

plt.savefig("problem_5_output.pdf", bbox_inches="tight")