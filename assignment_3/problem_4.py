#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:30:01 2024

@author: Rabin Meetarbhan
"""

import sys
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import aux

"""
Calculates the sample CDF of a set of data.

Adapted from https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
"""
def sample_cdf(x):
    x_len = np.size(x)
    
    return 1.0 * np.arange(x_len) / (x_len - 1)

data = np.loadtxt(sys.argv[1])

length, columns = data.shape

if columns < 2:
    raise ValueError("Supplied data file must contain at least two columns")

data1 = data[:,0]
data2 = data[:,1]

cdf1 = sample_cdf(data1)
cdf2 = sample_cdf(data2)

result = ks_2samp(data1, data2)

# === Output ===

print(f"KS Distance: {result.statistic}")
print(f"Location: {result.statistic_location}")
print(f"p-value: {result.pvalue}", end=" ")

if (result.pvalue < 0.05):
    print("(Likely not from the same distribution)")
else:
    print("(Likely from the same distribution)")

plt.figure(figsize=(6, 4), dpi=80)
plt.plot(np.sort(data1), cdf1, color="green")
plt.plot(np.sort(data2), cdf2, color="orange")
plt.xlim(left=-10, right=10)
plt.xlabel("$y$")
plt.ylabel("CDF")
plt.title("Comparison of the CDF of Two Datasets")
plt.rc('axes', axisbelow=True)
plt.grid()
plt.tight_layout()
plt.savefig("problem_4_output.pdf")