#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:00:24 2024

@author: Rabin Meetarbhan
"""

from math import factorial
import numpy as np
from numpy.linalg import eigh
from numpy.polynomial.hermite import hermvander
import matplotlib.pyplot as plt
import matplotlib.animation as animation

bsize = 100
# One extra level needs to be generated for the seventh to be stable for some reason...
# Done in code automatically
n = 7
x_ext = 5

npts = 1000
x = np.linspace(-x_ext, x_ext, npts)
dt = 0.01

# === Setting up operators in the Hermite basis ===

hamilt = np.diag(0.5 + np.arange(bsize))

x_sq = np.copy(hamilt)
p_sq = np.copy(hamilt)

for k in range((n + 1) - 2):
    u = 0.5 * np.sqrt((k + 1) * (k + 2))
    
    x_sq[k, k + 2] = u
    x_sq[k + 2, k] = u
    p_sq[k, k + 2] = -u
    p_sq[k + 2, k] = -u

basis_mat = hermvander(x, bsize - 1)
weights = np.exp(-0.5 * x * x) / (np.pi ** 0.25)

for i in range(n + 1):
    basis_mat[:, i] *= weights / np.sqrt(2.0 ** i * factorial(i))
    
# === Quartic oscillator ===

hamilt = 0.5 * p_sq + 0.25 * x_sq @ x_sq

energies, e_vecs = eigh(hamilt)

sstate = basis_mat @ e_vecs
pr = np.abs(sstate) ** 2
evo = np.zeros(bsize, dtype="complex")

# Only the first seven rows are filled out, since only they are being plotted.
# Can plot more or less stationary states by changing n.
for i in range(n + 1):
    evo[i] = np.exp(-1.0j * energies[i] * dt)

# === Plot setup ===

# Visual scaling factor for wavefunctions on plot
sf = 0.5

fig, ax = plt.subplots(figsize=(5, 10), dpi=90)

re = []
im = []

for i in range(n):
    replt, = ax.plot(x, energies[i] + sf * np.real(sstate[:, i]).T, color="red")
    implt, = ax.plot(x, energies[i] + sf * np.imag(sstate[:, i]).T, color="blue")
    re.append(replt)
    im.append(implt)

p = []

for i in range(n):
    p.append(plt.fill_between(x, energies[i] + sf * pr[:, i], energies[i],
                              color="grey", alpha=0.5, linewidth=0.0))

ax.set_xlim([-x_ext, x_ext])
ax.set_ylim([energies[0] - 0.5, energies[n - 1] + 0.5])
ax.set_yticks(np.arange(energies[0], energies[n], 1))

ax.set_xlabel(r"$x$")
ax.set_ylabel("Energy")

plt.tight_layout()

# === Animation ===

def tick(i):
    global sstate
    global pr
    
    sstate = evo * sstate
    pr = np.abs(sstate) ** 2

def animstep(i):
    tick(i + 1)
    
    for i in range(n):
        vertices = p[i].get_paths()[0].vertices
        vertices[1:n + 1, 1] = pr[:n, i]
        
        re[i].set_data(x, energies[i] + sf * np.real(sstate[:, i]))
        im[i].set_data(x, energies[i] + sf * np.imag(sstate[:, i]))
    
fps = 120

print("Drawing animation...")

anim = animation.FuncAnimation(fig, animstep, frames=1000, interval=1)
writer = animation.PillowWriter(fps=fps)
anim.save("problem_3_output.gif", writer=writer)

print("Done")