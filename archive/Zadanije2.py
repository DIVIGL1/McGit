# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:45:20 2016

@author: McSim
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

def func(x):
    y=np.sin(x/5)*np.exp(x/10)+5*np.exp(-x/2)
    return y

x=np.arange(1,16,1)

quant_x=0
while quant_x>15 or quant_x<1: 
    quant_x=int(input("Укажите количество точек:"))

mat_x=[]
mat_y=[]
for i in range(1,quant_x+1):
    mat_x.append(int(input("Введите "+str(i)+"-е число:")))
    mat_y.append(func(mat_x[i-1]))
    
print("Обрабатываем значения х:")
print(mat_x)
print("Функция в этих точках принимает значения:")
print(mat_y)

mat_Х=np.matrix([[mat_x[r]**(c) for c in range(quant_x)] for r in range(quant_x)])

mat_a=scipy.linalg.solve(mat_Х,mat_y)

print("Найденные коэффициенты:")
print(mat_a)

def f_new(x):
    y1=0
    for i in range(quant_x):
        y1 += mat_a[i]*(x**i)
    return y1
y2=f_new(x)

plt.plot(x,func(x),x,f_new(x))
plt.show()