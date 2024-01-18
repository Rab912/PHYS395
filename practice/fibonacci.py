# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:16:11 2024

@author: Rabin
"""
c1 = 0
c2 = 1
c3 = 0
n = 10

for i in range(n):
    c3 = c2 + c1
    c1 = c2
    c2 = c3
    print(c2)
    