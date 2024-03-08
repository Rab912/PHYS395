#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:19:30 2024

@author: Rabin Meetarbhan
"""

import problem_2

def func(x):
    return (x * x - 1) ** 2 + x

xmin1 = problem_2.gssmin(func, -1.5, 2, -1)
xmin2 = problem_2.gssmin(func, -1.5, 2, 0.9)

print(f"Minimum of {func(xmin1)} at {xmin1}")
print(f"Minimum of {func(xmin2)} at {xmin2}")