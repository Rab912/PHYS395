#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:59:30 2024

@author: Rabin Meetarbhan
"""

import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import aux

n_iter = 10000

skew_arr = np.empty(n_iter)
lskew_arr = np.empty(n_iter)

kurt_arr = np.empty(n_iter)
lkurt_arr = np.empty(n_iter)

for i in range(n_iter):
    n_samples = 1024

    x = np.random.standard_normal(n_samples)
    y = x + np.square(x) / 6.0
    
    lm = aux.lmoments(y, 4)
    
    skew_arr[i] = skew(y)
    lskew_arr[i] = lm[2] / lm[1]
    
    kurt_arr[i] = kurtosis(y)
    lkurt_arr[i] = lm[3] / lm[1]
    
# === Output ===

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), dpi=80)
fig.suptitle("Comparison of the Sample Distributions of Several Estimators for the Sample PDF of $y$")

bin_centers, pdf = aux.density_hist(skew_arr, 50)
axs[0, 0].bar(bin_centers, pdf, align='center', width=(bin_centers[1] - bin_centers[0]) * 0.8, color="red")
axs[0, 0].set_xlabel("Skewness")
axs[0, 0].set_ylabel("PDF")
axs[0, 0].grid()

bin_centers, pdf = aux.density_hist(kurt_arr, 50)
axs[0, 1].bar(bin_centers, pdf, align='center', width=(bin_centers[1] - bin_centers[0]) * 0.8, color="red")
axs[0, 1].set_xlabel("Kurtosis")
axs[0, 1].set_ylabel("PDF")
axs[0, 1].grid()

bin_centers, pdf = aux.density_hist(lskew_arr, 50)
axs[1, 0].bar(bin_centers, pdf, align='center', width=(bin_centers[1] - bin_centers[0]) * 0.8, color="red")
axs[1, 0].set_xlabel("L-Skewness")
axs[1, 0].set_ylabel("PDF")
axs[1, 0].grid()

bin_centers, pdf = aux.density_hist(lkurt_arr, 50)
axs[1, 1].bar(bin_centers, pdf, align='center', width=(bin_centers[1] - bin_centers[0]) * 0.8, color="red")
axs[1, 1].set_xlabel("L-Kurtosis")
axs[1, 1].set_ylabel("PDF")
axs[1, 1].grid()

fig.tight_layout()
plt.savefig("problem_3_output_correct.pdf")