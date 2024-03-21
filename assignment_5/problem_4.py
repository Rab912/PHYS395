#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:32:21 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def time(angle, amp):
    """
    Returns the integrand of the angular position-energy integral.

    Here, the angular frequency of oscillations is normalized to 1.
    """
    
    energy = -np.cos(amp)
    
    return 1.0 / np.sqrt(2.0 * (energy + np.cos(angle)))

n = 100
amps = np.linspace(0.01, 0.99 * np.pi, n)

p = np.empty(n)
p_approx = 2 * np.pi * np.ones(n)
err = 0

for i in range(n):
    sol = quad(time, -amps[i], amps[i], args=(amps[i],))
    p[i] = 2 * sol[0]

for i in range(n):
    err = np.abs((p_approx[i] - p[i]) / p[i])
    
    if err > 0.1:
        break
        
print(f"Amplitude of {amps[i]} required for 10% difference ({err})")

fig, ax = plt.subplots(figsize=(5, 5), dpi=90)
ax.plot(amps, p, color="red", label="Exact")
ax.plot(amps, p_approx, color="blue", label="Harmonic Approximation")
ax.set_xlabel("Amplitude / rad")
ax.set_ylabel("Period / s")
ax.set_title("Period versus Amplitude for a Swinging Pendulum")
ax.set_xlim(0, np.pi)
ax.set_ylim(0)
ax.grid()
ax.legend(frameon=False)
plt.savefig("problem_4_output.pdf", bbox_inches="tight")