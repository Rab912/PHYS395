#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:16:49 2024

@author: Rabin Meetarbhan
"""

import numpy as np

n = 100                     # Number of samples
x_max = 5                   # Upper x bound
x_min = -5                  # Lower x bound

x = np.linspace(x_min, x_max, n)
y = np.exp(-0.5 * np.square(x))

with open("problem_1_output.txt", "w+") as txt:
    for i in range(len(x)):
        txt.write(f"{x[i]:<24} {y[i]}\n")       # f-string formatting to align columns

