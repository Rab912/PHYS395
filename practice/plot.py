# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:16:11 2024

@author: Rabin
"""

import numpy as np
from matplotlib import pyplot as plt

n = 100
o = 8

x = np.linspace(-1.0, 1.0, n)

for i in range(o):
    plt.plot(x, np.cos(i * np.arccos(x)))

plt.show()
