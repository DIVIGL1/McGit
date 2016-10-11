# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 22:24:41 2016

@author: McSim
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import operator

def save_answer(sum, nr):
    with open("RES-W4-ans"+str(nr)+".txt", "w") as fout:
        fout.write(str(sum))

X = pd.read_csv('close_prices.csv', index_col='date')

pca = PCA(n_components=10)
pca.fit(X)
weights = pca.explained_variance_ratio_
print(weights)
sumx = 0
counter = 0
for x in weights:
    counter += 1
    if sumx < 0.90:
        sumx += x
    else:
        print(counter-1)
        break

save_answer(counter-1, 1)
    
X1 = pca.transform(X)

Y = pd.read_csv('djia_index.csv', index_col='date')

corr1 = np.corrcoef(X1[:,0], Y['^DJI'].values)

save_answer(round(corr1[0,1],2), 2)

num, maxnum = max(enumerate(pca.components_[0]), key=operator.itemgetter(1))
save_answer(X.columns[num], 3)