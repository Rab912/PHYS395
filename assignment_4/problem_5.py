#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:35:18 2024

@author: Rabin Meetarbhan
"""

import numpy as np
import problem_4 as p4
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit

def basis_fn(x, degree):
    freq = np.arange(0, degree + 1, 1)
    
    # In case of single-point evaluations
    if not isinstance(x, np.ndarray):
        return np.cos(2 * np.pi * x * freq)
        
    m = degree + 1
    n = len(x)
    fn = np.empty((m, n))
        
    for i in range(m):
        fnx = np.cos(2 * np.pi * x * freq[i])
        fn[i, :] = fnx
            
    return fn

data = p4.load_data_stdin()
x = data[:, 0]
y = data[:, 1]
errs = np.ones(len(y))

c = [0.8, 1.5, 0.3, 0, -5]
dof = len(x) - len(c)

popt, pcov, chisq = p4.lmexpfit(basis_fn, x, y, c, errs)

print(f"chisq = {chisq}, dof = {dof}")
print(f"popt = {popt}")
print(f"pcov = {pcov}")

x_fit = np.linspace(x[0], x[-1], 1000)
y_fit = p4._tfunc(x_fit, basis_fn, popt)

plt.figure(figsize=(8, 6), dpi=80)
plt.plot(x, y, ".", markersize=1, color="black")
label = r"$\alpha$ =" + f"{len(c) - 2}" + "\n$\chi^2$ =" + f"{chisq:.3f}" + f"\ndof = {dof}"
plt.plot(x_fit, y_fit, color="red", label=label)
plt.xlim(0, 1)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Plot of $y$ versus $x$")
plt.grid()
plt.legend()
plt.savefig("problem_5_output.pdf")

# curve_fit() gives pretty similar results

#def model(x, c0, c1, c2, c3, c4):
#    return np.exp(c0 + c1 * np.cos(2 * np.pi * x) +
#                  c2 * np.cos(4 * np.pi * x) + c3 * np.cos(6 * np.pi * x)) + c4
#
#popt, pcov = curve_fit(model, x, y, p0=c, sigma=errs, absolute_sigma=True)
#
#print("\ncurve_fit():")
#print(f"popt = {popt}")
#print(f"pcov = {pcov}")