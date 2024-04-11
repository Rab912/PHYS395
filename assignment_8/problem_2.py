#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:14:59 2024

@author: Rabin Meetarbhan
"""

from math import factorial
import numpy as np
from numpy.linalg import eigh, pinv
from numpy.polynomial.hermite import hermvander
import matplotlib.pyplot as plt
import matplotlib.animation as animation

npts = 1000
x_ext = 5
x = np.linspace(-x_ext, x_ext, npts)
dt = 0.01

# This value causes some calculations for the Hermite basis weights to reach
# close to the maximum floating point representation, so it should not be
# increased. There are noticeable wavy effects in the superposition (even though
# problem 1's output did not have these) because of the error induced in
# projecting the wavefunction onto a finite Hermite basis.
bsize = 150

# Can change this to any function
wave_func = np.exp(-0.5 * (x - 1) ** 2)

# === Setting up operators in the Hermite basis ===

hamilt = np.diag(0.5 + np.arange(bsize))

x_sq = np.copy(hamilt)
p_sq = np.copy(hamilt)

for k in range(bsize - 2):
    u = 0.5 * np.sqrt((k + 1) * (k + 2))
    
    x_sq[k, k + 2] = u
    x_sq[k + 2, k] = u
    p_sq[k, k + 2] = -u
    p_sq[k + 2, k] = -u

basis_mat = hermvander(x, bsize - 1)
weights = np.exp(-0.5 * x * x) / (np.pi ** 0.25)

for i in range(bsize):
    basis_mat[:, i] *= weights / np.sqrt(2.0 ** i * factorial(i))

coeffs = pinv(basis_mat) @ wave_func

# === Harmonic oscillator ===

hamilt = 0.5 * p_sq + 0.5 * x_sq

energies, e_vecs = eigh(hamilt)

sstate = basis_mat @ e_vecs
pr = np.abs(wave_func) ** 2
evo = np.zeros(bsize, dtype="complex")

for i in range(bsize):
    evo[i] = np.exp(-1.0j * energies[i] * dt)

# === Plot setup ===

fig, ax = plt.subplots(figsize=(5, 4), dpi=90)

re, = ax.plot(x, np.real(wave_func), color="red")
im, = ax.plot(x, np.imag(wave_func), color="blue")

p = plt.fill_between(x, pr, color="grey", alpha=0.5, linewidth=0.0)

ax.set_xlim([-x_ext, x_ext])
ax.set_ylim([-1.1, 1.1])

ax.set_xlabel(r"$x$")

plt.tight_layout()

# === Animation ===

def tick(i):
    global sstate
    global wave_func
    global pr
    
    sstate = evo * sstate
    wave_func = np.sum(coeffs * sstate, axis=1)
    pr = np.abs(wave_func) ** 2

def animstep(i):
    tick(i + 1)

    vertices = p.get_paths()[0].vertices
    vertices[1:npts + 1, 1] = pr
        
    re.set_data(x, np.real(wave_func))
    im.set_data(x, np.imag(wave_func))
    
fps = 120

print("Drawing animation...")

anim = animation.FuncAnimation(fig, animstep, frames=1000, interval=1)
writer = animation.PillowWriter(fps=fps)
anim.save("problem_2_output.gif", writer=writer)

print("Done")