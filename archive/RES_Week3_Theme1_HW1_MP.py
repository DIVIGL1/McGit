# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 00:39:47 2016

@author: McSim
"""

import pandas as pd
from sklearn.svm import SVC

columns = ["class","feature1","feature2"]

df_train = pd.read_csv('svm-data.csv',names=columns , index_col=None)
y_train = df_train["class"].values
X_train = df_train.drop("class", axis=1)

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#score_perceptron = metrics.accuracy_score(y_test, y_pred)
print('Indexes: ', clf.support_)