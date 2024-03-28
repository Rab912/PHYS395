#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:27:17 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt
import aux

def evolve_system(func, state0, t):
    """
    Function to automatically generate an array of states at certain times.
    """
    
    n = len(t)
    dt = (t[-1] - t[0]) / n
    state_arr = np.empty((len(state0), n))    
    
    state = np.copy(state0)
    
    for i in range(n):
        state_arr[:, i] = state
        state = aux.gaussleg8(func, state, dt)
        
    return state_arr

dt = 0.01
t = np.arange(0, 100, dt)

fig, ax = plt.subplots(figsize=(5, 5), dpi=90)

# ===

state0 = (2 * np.pi / 3, 2 * np.pi / 3, 0, 0)
a1_arr, a2_arr, _, _ = evolve_system(aux.double_pendulum, state0, t)

ax.plot(t, a1_arr, color="red", label=r"$\theta_1$, $\theta_2$ = $2\pi/3$")
ax.plot(t, a2_arr, color="crimson")

fig.text(0.85, 0.73, r"$\theta_1$")
fig.text(0.85, 0.28, r"$\theta_2$")

# ===

state0 = (2 * np.pi / 3 + 1e-6, 2 * np.pi / 3 + 1e-6, 0, 0)
a1_arr, a2_arr, _, _ = evolve_system(aux.double_pendulum, state0, t)

ax.plot(t, a1_arr, color="lime", label=r"$\theta_1$, $\theta_2$ = $2\pi/3 + 10^{-6}$")
ax.plot(t, a2_arr, color="limegreen")

fig.text(0.85, 0.51, r"$\theta_2$")

# ===

state0 = (2 * np.pi / 3 - 1e-6, 2 * np.pi / 3 - 1e-6, 0, 0)
a1_arr, a2_arr, _, _ = evolve_system(aux.double_pendulum, state0, t)

ax.plot(t, a1_arr, color="blue", label=r"$\theta_1$, $\theta_2$ = $2\pi/3 - 10^{-6}$")
ax.plot(t, a2_arr, color="darkblue")

fig.text(0.85, 0.15, r"$\theta_2$")

ax.set_xlabel("Time")
ax.set_ylabel("Angular Displacement / rad")

ax.set_xlim(0, 110)
ax.set_ylim(-42, 12)

ax.spines[['right', 'top']].set_visible(False)
for axis in ['left', 'bottom']:
    ax.spines[axis].set_linewidth(0.5)

ax.legend()

plt.savefig("problem_3_output.pdf", bbox_inches="tight")