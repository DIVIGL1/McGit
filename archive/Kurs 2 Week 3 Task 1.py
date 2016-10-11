# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:41:05 2016

@author: McSim
"""

from sklearn import cross_validation, datasets, grid_search, linear_model, metrics

import numpy as np
import pandas as pd

iris = datasets.load_iris()

train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(iris.data, iris.target, 
                                                                                     test_size = 0.3,random_state = 0)
                                                                                     
classifier = linear_model.SGDClassifier(random_state = 0)

print(classifier.get_params().keys())

parameters_grid = {
    'loss' : ['hinge', 'log', 'squared_hinge', 'squared_loss'],
    'penalty' : ['l1', 'l2'],
    'n_iter' : np.arange(5,10),
    'alpha' : np.linspace(0.0001, 0.001, num = 5),
}

cv = cross_validation.StratifiedShuffleSplit(train_labels, n_iter = 10, test_size = 0.2, random_state = 0)

grid_cv = grid_search.GridSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv)

print("\n", grid_cv)

grid_cv.fit(train_data, train_labels)

print("\n", grid_cv)

print("\n", grid_cv.best_estimator_)

print(grid_cv.best_score_)
print(grid_cv.best_params_)

print(grid_cv.grid_scores_[:10])

randomized_grid_cv = grid_search.RandomizedSearchCV(classifier, parameters_grid, scoring = 'accuracy', cv = cv, n_iter = 20, 
                                                   random_state = 0)
                                                   
randomized_grid_cv.fit(train_data, train_labels)

print(randomized_grid_cv.best_score_)
print(randomized_grid_cv.best_params_)