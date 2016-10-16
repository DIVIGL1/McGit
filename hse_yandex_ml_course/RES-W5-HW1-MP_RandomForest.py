# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:31:54 2016

@author: McSim
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score

df = pd.read_csv('abalone.csv', index_col=None, sep=",")

df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = df
y = X.Rings
X = X.drop('Rings', axis=1)

kf = KFold(X.shape[0], n_folds=5, random_state=1, shuffle=True)

r2_scores = []
r2_cv_means = []

for i in range(50):
    clf = RandomForestRegressor(n_estimators=i+1, random_state=1)
    clf.fit(X, y)
    r2_cv_means.append(np.mean(cross_val_score(clf, X, y, cv=kf, scoring='r2')))
    y_predicted = clf.predict(X)
    r2_scores.append(r2_score(y, y_predicted))
    
for i,j in enumerate(r2_scores):
    if j > 0.52:
        print(i+1)
        break
    
for i,j in enumerate(r2_cv_means):
    if j > 0.52:
        print(i+1)
        break
    
# Прямое использование случайного леса даёт уже со второго дерева хороший результат.
# При применении KFold качество резко падает. Почему?
# Предполагаю, что для случайного леса это лишний шаг - модель и так берёт случайный набор.
