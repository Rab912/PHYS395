#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:05:46 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.integrate import quad

def time(x):
    """
    Returns the integrand of the position-energy integral.
    """
    
    energy = 0.25 # E = (1/4) * lambda * amplitude ^ 4
    
    return 1.0 / np.sqrt(2.0 * energy - 0.5 * x ** 4)

# Moving from a displacement of -1 to 1 takes a half-period to do
sol = quad(time, -1, 1, epsabs=1E-16)

p = 2 * sol[0]
perr = 2 * sol[1]

# This method cannot go down to the same tolerance as by integrating the equation of motion
print(f"Period = {p} with error {perr} (quadrature integration)")