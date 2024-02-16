#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:19:43 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt
import aux

n = 5000

x = np.random.standard_normal(n)
y = x + np.square(x) / 6.0

y_bin_centers, y_pdf = aux.density_hist(y, 50)

# === Output ===

plt.figure(figsize=(6, 4), dpi=80)
plt.bar(y_bin_centers, y_pdf, align='center', width=(y_bin_centers[1] - y_bin_centers[0]) * 0.8,
        color="orange")
plt.xlabel("$y$")
plt.ylabel("PDF")
plt.title("Estimate of Probability Density of $y = x^2 + x / 6$")
plt.rc('axes', axisbelow=True)
plt.grid()
plt.tight_layout()
plt.savefig("problem_2_output.pdf")