#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 01:59:49 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import aux

# The far-away boundary condition is zero only for energy eigenvalues, so need
# to find the roots of this boundary condition function to get these energies.

def bc(energy, even, inf):
    """
    Determines the far-away boundary condition corresponding to an initial state, by
    "shooting out" a solution.
    
    Sets one boundary condition at the origin as func(0) = 0.
    """
    
    func = lambda x, state : aux.stationary(x, state, aux.qpotential, energy)
    
    # At the origin, even solutions have cosine-looking behavior, and odd solutions
    # have sine-looking behavior.
    state = (1, 0) if even else (0, 1)
    
    sol = solve_ivp(func, [0, inf], state)
    
    return sol.y[0, -1]

def gen_wf(x, state0, potential, energy):
    """
    Generates a stationary wavefunction with the given energy level, potential, and state
    at the origin (condensed version of problem 1).
    """
    
    wf = lambda x, state : aux.stationary(x, state, potential, energy)

    state = solve_ivp(wf, (x[0], x[-1]), state0, t_eval=x, atol=1E-12)
    
    return state.y[0]

state0 = (0, 1)
n = 1000
x = np.linspace(0, 10, n)

nlevels = 10


# These levels were manually tested for a sign change far away
#0: 0.41 - 0.43
#1: 1.4 - 1.6
#2: 2.8 - 3
#3: 4.5 - 4.7
#4: 6.3 - 6.5
#5: 8.3 - 8.5
#6: 10.4 - 10.6
#7: 12.6 - 12.8
#8: 14.9 - 15.1
#9: 17.3 - 17.5
energies_expected = [0.42, 1.5, 2.9, 4.6, 6.4, 8.4, 10.5, 12.7, 15, 17.4]

energies = np.empty(nlevels)
wave_funcs = np.empty((nlevels, n))
probabilities = np.empty((nlevels, n))

for level in range(nlevels):
    a = energies_expected[level] - 0.1
    b = energies_expected[level] + 0.1
    
    state0 = (1, 0) if level % 2 == 0 else (0, 1)
    
    if level % 2 == 0:
        energies[level] = bisect(bc, a, b, args=(True, 10))
        
    else:
        energies[level] = bisect(bc, a, b, args=(False, 10))

    wf = gen_wf(x, state0, aux.qpotential, energies[level])
    
    # Integrating over the entire interval to get the normalization factors will
    # not work due to wavefunctions still diverging. A range where it "acts normal"
    # was found by plotting the initial results and latter used here
    nf = simpson(wf[:400] ** 2, x[:400])
    
    wave_funcs[level, :] = wf / np.sqrt(nf)
    probabilities[level, :] = wf * wf / nf

fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 7), dpi=90)
plt.subplots_adjust(wspace=0.05)

for level in range(nlevels):
    # Not plotted to scale
    # The energies found by bisection are not accurate enough to keep the wavefunctions
    # finite for even a short time. They all start diverging at roughly x = 4. The range
    # for plotting was kept the same to be consistent.
    ax[0].plot(x, level + 0.3 * wave_funcs[level, :], color="black")
    ax[1].plot(x, level + 0.8 * probabilities[level, :], color="red")
    ax[0].text(3, level + 0.35, r"$E$ = " + f"{energies[level]:.5f}", fontsize=8)

ax[0].set_yticks(np.arange(0, 10, 1))
ax[0].set_ylim(-0.5, 10)
ax[0].set_xlim(0, 5)
ax[0].set_xlabel(r"$x$")
ax[0].set_title("Probability Amplitude")

ax[1].set_xlim(0, 5)
ax[1].set_xlabel(r"$x$")
ax[1].set_title("Probability Density")

ax[0].set_ylabel("Energy Number")

plt.savefig("problem_3_output.pdf", bbox_inches="tight")