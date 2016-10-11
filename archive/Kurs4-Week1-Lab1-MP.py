# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 23:45:22 2016

@author: McSim
"""

from sklearn import cross_validation, datasets, linear_model, metrics
from matplotlib import pyplot as P 

import numpy as np

blobs = datasets.make_blobs(300, centers = 2, cluster_std = 6, random_state=1)

P.figure()
cmap = 'autumn' # P.matplotlib.cm.jet 
X = [x[0] for x in  blobs[0]]
Y = [x[1] for x in  blobs[0]]
P.scatter(X, Y, cmap=cmap, c = blobs[1], s=100)