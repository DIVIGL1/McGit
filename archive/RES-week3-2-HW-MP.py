# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:08:01 2016

@author: McSim
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

columns = ['class', 'feature1', 'feature2']
df = pd.read_csv('data-logistic.csv',names=columns , index_col=None)
df_class = df["class"].values
df_data = df.drop("class", axis=1)

k = 0.1 # длина шага
C = 10 # Коэффициент регуляризации
n_steps = 10000 # Количество итераций
min_euclidean_distance = 1e-5
l = len(df_data)

## С регуляризацией
w1 = 0
w2 = 0
w = np.array((w1,w2))
for iter in range(n_steps):
    w1 += k * 1/l * sum(x[0]*x[1]*(1-1/(1+np.exp(-x[0]*(w1*x[1]+w2*x[2])))) for x in df.as_matrix()) - k*C*w1
    w2 += k * 1/l * sum(x[0]*x[2]*(1-1/(1+np.exp(-x[0]*(w1*x[1]+w2*x[2])))) for x in df.as_matrix()) - k*C*w2
    if np.linalg.norm(w-np.array((w1,w2))) < min_euclidean_distance:
        break
    w = np.array((w1,w2))
y_pred = np.array([1/(1+np.exp(-w1*x[1]-w2*x[2])) for x in df.as_matrix()])
print('AUC-ROC: ', roc_auc_score(df_class, y_pred))
print('number of iterations: ', iter)
print('weights: ', w1, w2)
print('The last euclidean distance: ', np.linalg.norm(w-np.array((w1,w2))))

## Без регуляризации
w1 = 0
w2 = 0
w = np.array((w1,w2))
for iter in range(n_steps):
    w1 += k * 1/l * sum(x[0]*x[1]*(1-1/(1+np.exp(-x[0]*(w1*x[1]+w2*x[2])))) for x in df.as_matrix())
    w2 += k * 1/l * sum(x[0]*x[2]*(1-1/(1+np.exp(-x[0]*(w1*x[1]+w2*x[2])))) for x in df.as_matrix())
    if np.linalg.norm(w-np.array((w1,w2))) < min_euclidean_distance:
        break
    w = np.array((w1,w2))
y_pred = np.array([1/(1+np.exp(-w1*x[1]-w2*x[2])) for x in df.as_matrix()])
print('AUC-ROC: ', roc_auc_score(df_class, y_pred))
print('number of iterations: ', iter)
print('weights: ', w1, w2)
print('The last euclidean distance: ', np.linalg.norm(w-np.array((w1,w2))))
