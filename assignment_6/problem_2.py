#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:07:16 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aux

DRAW_ANIM = False

state = (np.pi / 3, -np.pi / 3, 0, 0)

n = 5000
t = np.linspace(0, 100, n) # Angular frequency is normalized to 1, so t = 100/1
dt = (t[-1] - t[0]) / n

a1_arr = np.empty(n)
a2_arr = np.empty(n)
energy_arr = np.empty(n)

for i in range(n):
    a1_arr[i] = state[0]
    a2_arr[i] = state[1]
    energy_arr[i] = aux.energy(state)
    
    state = aux.gaussleg8(aux.double_pendulum, state, dt)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=90)

ax[0].plot(t, a1_arr, color="red", label=r"$\theta_1$")
ax[0].plot(t, a2_arr, color="blue", label=r"$\theta_2$")

ax[0].set_xlabel("Time")
ax[0].set_ylabel("Angular Displacement / rad")

ax[1].plot(t, energy_arr, color="orange")

ax[1].set_xlabel("Time")
ax[1].set_ylabel("Energy")

ax[0].spines[['right', 'top']].set_visible(False)
ax[1].spines[['right', 'top']].set_visible(False)
for axis in ['left', 'bottom']:
    ax[0].spines[axis].set_linewidth(0.5)
    ax[1].spines[axis].set_linewidth(0.5)

ax[0].legend()

plt.savefig("problem_2_output.pdf", bbox_inches="tight")

# Optional animation

if DRAW_ANIM:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
    
    # Drawing outer boundaries of the reachable space of each pendulum rod
    ax.add_patch(plt.Circle((0, 0), 1.0, color='r', fill=False))
    ax.add_patch(plt.Circle((0, 0), 2.0, color='r', fill=False))
    
    ax.set_aspect('equal')
    plt.xlim([-2.2,2.2])
    plt.ylim([-2.2,2.2])
    
    # Initial state - the one used above
    state = (np.pi / 3, -np.pi / 3, 0, 0)
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
    anim.save("problem_2_output_opt.gif", writer=writer)
    
    print("Done")