# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:18:15 2016

@author: McSim
"""

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
 
from sklearn.datasets import load_boston
boston = load_boston()
rf = RandomForestRegressor()
rf.fit(boston.data[:300], boston.target[:300])

instances = boston.data[[300, 309]]

print("Instance 0 prediction:", rf.predict(instances[0]))
print("Instance 1 prediction:", rf.predict(instances[1]))

prediction, biases, contributions = ti.predict(rf, instances)

for i in range(len(instances)):
    print("Instance", i)
    print("Bias (trainset mean)", biases[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions[i], 
                                 boston.feature_names), 
                             key=lambda x: -abs(x[0])):
        print(feature, round(c, 2))
    print("-"*20)
    
print(prediction)
print(biases + np.sum(contributions, axis=1))