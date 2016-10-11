# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 00:37:16 2016

@author: McSim
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

#df = pn.DataFrame.from_csv(path="checkins.csv", header=0, sep=';')
df = pd.DataFrame.from_csv(path="checkins.dat", header=0, sep='|')
df.drop(df.index[[0]], inplace=True)
df.rename(columns=lambda x: x.strip(), inplace=True)
df['latitude'] = df['latitude'].str.replace(" ","")
df = df[df.latitude != '']

subset = df[:100000]
subset = subset.drop(['user_id','venue_id','created_at'], 1)

bandwidth = 0.1
ms = MeanShift(bandwidth=bandwidth)
ms.fit(subset)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels, return_counts=True)
clusters = np.c_[cluster_centers, labels_unique[:][1]]

print("number of estimated clusters : %d" % len(labels_unique[0]))

min_obj_quantity = 15
clusters = clusters[clusters[:,2] > min_obj_quantity]

print("Из них {0} имеют более {1} объектов".format(len(clusters),min_obj_quantity))

offices = np.array([[33.751277, -118.188740, 'Los Angeles'], [25.867736, -80.324116, 'Miami'],
                     [51.503016, -0.075479, 'London'], [52.378894, 4.885084,'Amsterdam'],
                     [39.366487, 117.036146, 'Beijing'], [-33.868457, 151.205134, 'Sydney']])

for i in range(len(offices)):
    clusters =  np.c_[clusters, np.zeros((len(clusters),1))]
    for j in range(len(clusters)):
        clusters[j][3+i] = ((clusters[j][0] - float(offices[i][0]))**2 + ((clusters[j][1] - float(offices[i][1]))**2))**0.5

clusters =  np.c_[clusters, np.zeros((len(clusters),1))]
for j in range(len(clusters)):
    clusters[j][3+i+1] = min(clusters[j][3:-2])

sorted = clusters[clusters[:,9].argsort()]
print(sorted[:,:20])
print('Answer:')
print(sorted[0,0])
print(sorted[0,1])