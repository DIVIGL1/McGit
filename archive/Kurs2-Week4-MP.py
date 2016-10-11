# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 21:21:16 2016

@author: McSim
"""

from sklearn import datasets, tree, linear_model, cross_validation, metrics 

import numpy as np
import pandas as pd

boston = datasets.load_boston()

test_data_quota = 0.25

full_size = boston.data.shape[0]
test_size = int(full_size*test_data_quota + 0.5)
X_train, X_test, y_train, y_test = boston.data[:(full_size-test_size)], boston.data[(full_size-test_size):], boston.target[:(full_size-test_size)], boston.target[(full_size-test_size):]
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(boston.data, boston.target, test_size = test_data_quota, random_state=0)

def answer(a,y):
    return (y-a)
    
n_trees = range(50)
scores = list()
start_coef = 0.9
base_algorithms_list = list()
coefficients_list = list()

def gbm_predict(X):
    return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_algorithms_list, coefficients_list)]) for x in X]
    
for n_tree in n_trees:
    regressor = tree.DecisionTreeRegressor(random_state=42, max_depth=5)
    regressor.fit(X_train, answer(gbm_predict(X_train), y_train))
    base_algorithms_list = np.append(base_algorithms_list, regressor)
    coefficients_list = np.append(coefficients_list, start_coef) #/(1.0 + n_tree)
    scores.append(np.sqrt(metrics.mean_squared_error(y_test, gbm_predict(X_test))))

final_score = (metrics.mean_squared_error(y_test, gbm_predict(X_test)))**0.5

def write_answer_1(ans):
    with open("ans1.txt", "w") as fout:
        fout.write(str(ans))
        
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
linear_prediction = linear_regressor.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, linear_prediction))
print(rmse)