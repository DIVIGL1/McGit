# -*- coding: utf-8 -*-
"""
Created on Mon May 30 00:02:30 2016

@author: McSim
"""

#import numpy as np
import pandas as pd
#import urllib
from sklearn import cross_validation
#from sklearn import datasets
#from sklearn import svm 
#from sklearn import metrics
#from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import preprocessing
columns = ["class","alcohol","malic-acid","ash","alcalinity-of-ash","magnesium","total-phenols","flavanoids",
           "nonflavanoid-phenols","proanthocyanins","color-intensity","hue","OD280","proline"]
dataset = pd.read_csv('C:\\Users\\McSim\\Documents\\Python Scripts\\wine.data.txt',names=columns, index_col=None)
print(dataset.shape)
Y = dataset["class"].values
X = dataset.drop("class", axis=1)
X = preprocessing.scale(X)
kf = KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)

kMeans = list()
i = 0
while i < 50:
    i = i + 1  
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X, Y)
    #KNeighborsClassifier(...)
    #print(neigh.predict(X))
    print(str(i) + " : " + str(neigh.score(X, Y)))
    array = cross_validation.cross_val_score(estimator=neigh, X=X, y=Y, cv=kf, scoring='accuracy')
    m = array.mean()
    print("Accuracy: %0.2f (+/- %0.2f)" % (m, array.std() * 2))
    kMeans.append(m)
    
m = max(kMeans)
print(m)
 
scaler = preprocessing.StandardScaler().fit(X)


