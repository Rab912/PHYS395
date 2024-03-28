# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:45:24 2024

@author: Rabin Meetarbhan
"""

import numpy as np

def double_pendulum(state):    
    """
    A set of coupled differential equations representing the dynamic state of a double
    pendulum, including the angle and generalized momentum of each rod.
    
    The mass and length of each rod and gravity are all normalized to 1.
    """
    
    a1, a2, p1, p2 = state
    
    cosdiff = np.cos(a1 - a2)
    sindiff = np.sin(a1 - a2)
    
    c = 1 / (16.0 / 9 - cosdiff ** 2)
    
    ad1 = c * (2 * p1 / 3 - cosdiff * p2)
    ad2 = c * (8 * p2 / 3 - cosdiff * p1)
    
    pd1 = -3 * np.sin(a1) - sindiff * ad1 * ad2
    pd2 = -np.sin(a2) + sindiff * ad1 * ad2
    
    return ad1, ad2, pd1, pd2

def energy(state):
    """
    Calculates the total energy of a double pendulum when in a particular state, relative
    to the "up" position.
    
    The mass and length of each rod and gravity are all normalized to 1.
    """
    
    a1, a2, p1, p2 = state
    
    cosdiff = np.cos(a1 - a2)
    
    c = 1 / (16.0 / 9 - cosdiff ** 2)
    
    ad1 = c * (2 * p1 / 3 - cosdiff * p2)
    ad2 = c * (8 * p2 / 3 - cosdiff * p1)
    
    ke = 4 * (ad1 ** 2) / 3 + (ad2 ** 2) / 3 + ad1 * ad2 * cosdiff
    pe = -3 * np.cos(a1) - np.cos(a2)
    
    return ke + pe

# 8'th order Butcher tableau (in quad precision)
a4 = np.array([
	 0.869637112843634643432659873054998518E-1,
	-0.266041800849987933133851304769531093E-1,
	 0.126274626894047245150568805746180936E-1,
	-0.355514968579568315691098184956958860E-2,
	 0.188118117499868071650685545087171160E0,
	 0.163036288715636535656734012694500148E0,
	-0.278804286024708952241511064189974107E-1,
	 0.673550059453815551539866908570375889E-2,
	 0.167191921974188773171133305525295945E0,
	 0.353953006033743966537619131807997707E0,
	 0.163036288715636535656734012694500148E0,
	-0.141906949311411429641535704761714564E-1,
	 0.177482572254522611843442956460569292E0,
	 0.313445114741868346798411144814382203E0,
	 0.352676757516271864626853155865953406E0,
	 0.869637112843634643432659873054998518E-1
]).reshape([4,4])

b4 = np.array([
	 0.173927422568726928686531974610999704E0,
	 0.326072577431273071313468025389000296E0,
	 0.326072577431273071313468025389000296E0,
	 0.173927422568726928686531974610999704E0
])

def gaussleg8(func, y, dt):
    """
    Computes an 8'th order Gauss-Legendre step.
    
    Taken from the course GitHub repository.
    """
    
    n = len(y)
    g = np.zeros([4, n])
    
    for k in range(16):
        g = np.matmul(a4, g)
        
        for i in range(4):
            g[i] = func(y + g[i] * dt)
            
    return tuple(y + np.dot(b4, g) * dt)