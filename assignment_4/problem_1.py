#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:19:50 2024

@author: Rabin Meetarbhan
"""

"""
Locates a root of the given function within the specified interval, if there are any.

Uses the Newton-Raphson method, so its derivative is required.
"""
def nrroot(fn, df, x1, x2, acc=1E-14, niter=100):
    root = 0.5 * (x1 + x2)

    for _ in range(niter):
        func = fn(root)
        deriv = df(root)
        dx = func / deriv
        root -= dx
        
        if (x1 - root) * (root - x2) < 0.0:
            raise ValueError("Root jumped out of bounds")
            
        if (abs(dx) < acc):
            return root
    
    raise ValueError("Reached maximum number of iterations")

def fn(x):
    return x * x * x - x + 0.25

def df(x):
    return 3 * x * x - 1

root1 = nrroot(fn, df, -1.5, -0.5)
root2 = nrroot(fn, df, 0.0, 0.5)
root3 = nrroot(fn, df, 0.5, 1.0)

print(f"x = {root1}, f(x) = {fn(root1)}")
print(f"x = {root2}, f(x) = {fn(root2)}")
print(f"x = {root3}, f(x) = {fn(root3)}")