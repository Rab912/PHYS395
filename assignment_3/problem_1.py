#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:55:22 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import matplotlib.pyplot as plt
import aux

"""
Returns an array containing the result performed on the first k elements of the given array stored in the k'th entry.

The given function must have only one required parameter and must return a scalar value.
"""
def evolve(x, func):
    return np.array([func(x[:i]) for i in range(1, n + 1)])

# Lambda to make lmoments() compatible with evolve()
lvariance = lambda x: aux.lmoments(x, 2)[1]

n = 8192
nspace = np.arange(1, n + 1, 1)

# === Gaussian Distribution ===

g_samples = np.random.standard_normal(n)

g_mean = evolve(g_samples, np.mean)
g_median = evolve(g_samples, np.median)
g_variance = evolve(g_samples, np.var)
g_lvariance = evolve(g_samples, lvariance)

# === Cauchy Distribution ===

c_samples = np.random.standard_cauchy(n)

c_mean = evolve(c_samples, np.mean)
c_median = evolve(c_samples, np.median)
c_variance = evolve(c_samples, np.var)
c_lvariance = evolve(c_samples, lvariance)

# === Output ===

print(f"Gaussian sample mean for {n} samples: {g_mean[-1]}")
print(f"Gaussian sample median for {n} samples: {g_median[-1]}")
print(f"Gaussian sample variance for {n} samples: {g_variance[-1]}")
print(f"Gaussian sample L-variance for {n} samples: {g_lvariance[-1]}")

print(f"Cauchy sample mean for {n} samples: {c_mean[-1]} (indefinite)")
print(f"Cauchy sample median for {n} samples: {c_median[-1]}")
print(f"Cauchy sample variance for {n} samples: {c_variance[-1]} (indefinite)")
print(f"Cauchy sample L-variance for {n} samples: {c_lvariance[-1]} (indefinite)")

fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, figsize=(8, 6), dpi=80)
fig.suptitle("Comparison of the Evolution of Several Estimators for Standard Gaussian and\nCauchy Distributions")

axs[0, 0].plot(nspace, g_mean, color="red", label="Gaussian")
axs[0, 0].plot(nspace, c_mean, color="blue", label="Cauchy (indefinite)")
axs[0, 0].set_xscale("log", base=2)
axs[0, 0].set_ylabel("Mean")
axs[0, 0].grid()
axs[0, 0].legend(prop={'size': 8})

axs[0, 1].plot(nspace, g_median, color="red", label="Gaussian")
axs[0, 1].plot(nspace, c_median, color="blue", label="Cauchy")
axs[0, 1].set_xscale("log", base=2)
axs[0, 1].set_ylabel("Median")
axs[0, 1].grid()
axs[0, 1].legend(prop={'size': 8})

axs[1, 0].plot(nspace, g_variance, color="red", label="Gaussian")
axs[1, 0].plot(nspace, c_variance, color="blue", label="Cauchy (indefinite)")
axs[1, 0].set_xscale("log", base=2)
axs[1, 0].set_xlabel("Sample size")
axs[1, 0].set_ylabel("Variance")
axs[1, 0].grid()
axs[1, 0].legend(prop={'size': 8})

axs[1, 1].plot(nspace, g_lvariance, color="red", label="Gaussian")
axs[1, 1].plot(nspace, c_lvariance, color="blue", label="Cauchy (indefinite)")
axs[1, 1].set_xscale("log", base=2)
axs[1, 1].set_xlabel("Sample size")
axs[1, 1].set_ylabel("L-Variance")
axs[1, 1].grid()
axs[1, 1].legend(prop={'size': 8})

fig.tight_layout()
plt.savefig("problem_1_output.pdf")