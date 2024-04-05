#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:52:19 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import aux

energy = 100
state0 = (1, -1)
dx = 0.01
x = np.arange(0, 15 + dx, dx)

# A different potential function and energy can be used here.
eqn = lambda x, state : aux.stationary(x, state, aux.hpotential, energy)

state_p = solve_ivp(eqn, (x[0], x[-1]), state0, t_eval=x, atol=1E-12).y[0]
state_n = solve_ivp(eqn, (-x[0], -x[-1]), state0, t_eval=-x, atol=1E-12).y[0]

# The errors in these should not be much larger than the tolerances given above.
# It is evident that the even and odd solutions asymptotically approach +inf
# and -inf, respectively, since the energy is not an eigenvalue.
sol_even = 0.5 * (state_p + state_n)
sol_odd = 0.5 * (state_p - state_n)

fig, ax = plt.subplots(figsize=(5, 5), dpi=90)

ax.plot(x, sol_even, color="red", label=r"$\psi_+(x)$")
ax.plot(x, sol_odd, color="blue", label=r"$\psi_-(x)$")
#ax.plot(x, state_p, color="green", label=r"$\psi(x)$")

ax.set_xlim(x[0], x[-1])
ax.set_xlabel("$x$")
ax.set_ylabel("Probability Amplitude")

ax.set_title("Even and Odd Wavefunctions for Harmonic\n" + r"Potential and $E$ = " + f"{energy}")

ax.legend()

plt.savefig("problem_1_output.pdf", bbox_inches="tight")