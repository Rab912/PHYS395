#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:37:53 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aux

fig, ax = plt.subplots(figsize=(5, 5), dpi=90)

# Drawing outer boundaries of the reachable space of each pendulum rod
ax.add_patch(plt.Circle((0, 0), 1.0, color='r', fill=False))
ax.add_patch(plt.Circle((0, 0), 2.0, color='r', fill=False))

ax.set_aspect('equal')
plt.xlim([-2.2,2.2])
plt.ylim([-2.2,2.2])

# Initial state - first joint is horizontal and second joint is vertically up
state = (0.5 * np.pi, np.pi, 0, 0)
a1, a2, _, _ = state

pend1, = ax.plot([0, np.sin(a1)],
                 [0, -np.cos(a1)],
                 "o-", color="blue", linewidth=7, ms=15)

pend2, = ax.plot([np.sin(a1), np.sin(a1) + np.sin(a2)],
                 [-np.cos(a1), -np.cos(a1) - np.cos(a2)],
                 "o-", color="blue", linewidth=7, ms=15)

#energy_prev = 0
energy0 = aux.energy(state)
de_text = fig.text(0.15, 0.85, f"dE = {0}")

# This time step keeps error in total energy below 10^-12.
dt = 1.0 / 50

# Can also display instantaneous energy change by uncommenting all lines relating
# to energy_prev, and replacing de with the commented definition
def animstep(i):
    global state
    #global energy_prev
    
    state = aux.gaussleg8(aux.double_pendulum, state, dt)
    
    a1, a2, _, _ = state
    
    pend1.set_data([0, np.sin(a1)],
                   [0, -np.cos(a1)])
    
    pend2.set_data([np.sin(a1), np.sin(a1) + np.sin(a2)],
                   [-np.cos(a1), -np.cos(a1) - np.cos(a2)])
    
    energy = aux.energy(state)
    
    #de = energy - energy_prev
    de = energy - energy0
    de_text.set_text(f"dE = {de:.3e}")
    
    #energy_prev = energy

fps = 60
frames = 1000

print("Drawing animation. This will take a moment...")

anim = animation.FuncAnimation(fig, animstep, frames=frames, interval=0.001 * frames / fps)
writer = animation.PillowWriter(fps=fps)
anim.save("problem_1_output.gif", writer=writer)

print("Done")