# -*- coding: utf-8 -*-
"""
Created on Tue May 31 01:43:26 2016

@author: McSim
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation
from sklearn.cross_validation import KFold
boston = load_boston()
X, y = boston.data, boston.target
X = preprocessing.scale(X)
kf = KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)
p_set = np.linspace(1, 10, num=200)
kMeans = list()
for i in p_set:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', 
                    leaf_size=30, p=i, metric='minkowski', metric_params=None, 
                    n_jobs=1)
    neigh.fit(X, y)
    array = cross_validation.cross_val_score(estimator=neigh, X=X, y=y, cv=kf, 
                                             scoring='mean_squared_error')
    m = array.mean()
    #print("Accuracy: %0.2f (+/- %0.2f)" % (m, array.std() * 2))
    kMeans.append(m)
    
m = max(kMeans)
print(m)

