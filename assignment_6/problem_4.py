#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:56:39 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import aux

n = 100

a1_init = np.linspace(-np.pi + 0.1, np.pi - 0.1, n)
a2_init = np.linspace(-np.pi + 0.1, np.pi - 0.1, n)

flipped_arr = np.full((n, n), 0)

dt = 0.01
t = np.arange(0, 100, dt) # Angular frequency is normalized to 1, so t = 100/1
nt = len(t)

print("Computing trajectories. This will take a very long time...")

for i in range(n):
    for j in range(n):
        state = (a1_init[i], a2_init[j], 0, 0)
        flipped = False
        
        # Integrator which stops when an angle exceeds pi in magnitude
        for k in range(nt):
            state = aux.gaussleg8(aux.double_pendulum, state, dt)
            angles = (state[0], state[1])
            max_angle = np.max(np.abs(angles))
            
            if max_angle > np.pi:
                flipped_arr[i, j] = 1
                flipped = True
                break
        
        if not flipped:
            flipped_arr[i, j] = max_angle / np.pi
            
fig, ax = plt.subplots(figsize=(5, 5), dpi=90)

im = ax.imshow(flipped_arr, extent=(a1_init[0], a1_init[-1], a2_init[0], a2_init[-1]),
          aspect="equal", cmap=cm.hot, norm=LogNorm())

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

ax.set_xlabel(r"$\theta_2$")
ax.set_ylabel(r"$\theta_1$")

plt.savefig("problem_4_output.pdf", bbox_inches="tight")