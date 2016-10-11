# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 20:31:34 2016

@author: McSim
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def write_answer(num, input_data):
    with open("MP-kNN-ans-"+str(num)+".txt", "w") as fout:
        fout.write(str(input_data))

ds = datasets.load_digits()

test_data_quota = 0.25

data_size = int(ds.data.shape[0]*(1-test_data_quota))
data_train, data_test, target_train, target_test = ds.data[:data_size], ds.data[data_size:], ds.target[:data_size], ds.target[data_size:]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(data_train, target_train)
ans = neigh.score(data_test, target_test)

write_answer(1, (1-ans))

cls = RandomForestClassifier(n_estimators=1000)
cls.fit(data_train, target_train)
ans = cls.score(data_test, target_test)

write_answer(2, (1-ans))

#В задании было предолжено самостоятельно реализолвать метод 1NN
#import numpy as np
#single_point = [3, 4]
#points = np.arange(20).reshape((10,2))
#
#dist = (points - single_point)**2
#dist = np.sum(dist, axis=1)
#dist = np.sqrt(dist)