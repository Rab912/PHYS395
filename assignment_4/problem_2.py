# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:02:37 2024

@author: Rabin Meetarbhan
"""

"""
Does a bracketed minimum search of a function by golden sections.

Takes an initial bracketing tuple (a, b, c), or an interval (a, c) which contains a local minimum if b is not supplied.
"""
def gssmin(func, a, c, b=None, tol=1E-8, return_interval=False):
    lr = 0.6180339887502366
    sr = 1.0 - lr
    
    if b is None:
        b = lr * a + sr * c
    
    d = 0.0

    if (abs(c - b) > abs(b - a)):
        d = lr * b + sr * c # or b + sr * (c - b)
        
    else:
        d = sr * a + lr * b
        b, d = d, b # Keeping b < d
        
    while abs(c - a) > tol * (abs(b) + abs(d)):
        fb = func(b)
        fd = func(d)
        
        # Narrowing the bracketing tuple while keeping a < b < d < c in either case
        if fd < fb:
            a = b
            b = d
            d = lr * d + sr * c
        else:
            c = d
            d = b
            b = sr * a + lr * b
    
    if not return_interval:
        if func(a) < func(c):
            return a
        
        else:
            return c
    
    else:
        return (a, c)