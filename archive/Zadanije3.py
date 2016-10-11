# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:45:20 2016

@author: McSim
"""

import numpy as np
from math  import sin, exp
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

func = lambda x: np.sin(x/5)*np.exp(x/10)+5*np.exp(-x/2)

x_min = 1
x_max = 30

#plt.plot(x,func(x))
#plt.show()

print("Ищем минимум стандартной функцией scipy.optimize.minimize:")
#for x0 in range(x_min, x_max+1):
#x0 = np.ndarray(shape=(1,), buffer=np.array([x for x in range(x_min,x_max+1)]), dtype=int)
x1 = np.arange(x_min,x_max+1,1)

#plt.plot(x1,func(x1))
#plt.show()

#print(minimize(func,x0, method='BFGS'))
print(minimize(func,30, method='BFGS'))

x2=[(x_min,x_max)]
print(differential_evolution(func,x2))

def h(x):
    return func(x).astype(int)
    
func1 = lambda x: np.int(np.sin(x/5)*np.exp(x/10)+5*np.exp(-x/2))
    
plt.plot(x1,func(x1),x1,h(x1))
plt.show()

print(minimize(h,30, method='BFGS'))
print(differential_evolution(h,x2))