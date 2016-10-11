# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:57:06 2016

@author: McSim
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

columns = ["class","feature1","feature2"]

df_train = pd.read_csv('perceptron-train.csv',names=columns , index_col=None)
y_train = df_train["class"].values
X_train = df_train.drop("class", axis=1)

df_test = pd.read_csv('perceptron-test.csv',names=columns , index_col=None)
y_test = df_test["class"].values
X_test = df_test.drop("class", axis=1)

#X = preprocessing.scale(X)

#from sklearn.linear_model import Perceptron
#X = np.array([[1,2],[3,4],[5,6]])
#y = np.array([0,1,0])
#clf = Perceptron()
#clf.fit(X,y)
#predictions = clf.predict(X)

#sklearn.metrics.accuracy_score
#sklearn.preprocessing.StandardScaler

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train = np.array([[100.0,2.0],[50.0,4.0],[70.0,6.0]])
#X_test = np.array([[90.0,1],[40.0,3],[60.0,4]])
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score_perceptron = metrics.accuracy_score(y_test, y_pred)
print('Accuracy of Perceptron with not scaled data: ', score_perceptron)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_scaled = Perceptron(random_state=241)
clf_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = clf_scaled.predict(X_test_scaled)
score_perceptron_scaled = metrics.accuracy_score(y_test, y_pred_scaled)
print('Accuracy of Perceptron with scaled data: ', score_perceptron_scaled)

print('Answer is ', score_perceptron_scaled - score_perceptron)