# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 18:43:38 2016

@author: McSim
"""

import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.weightstats import *

import sklearn
data = pd.DataFrame.from_csv('banknotes.txt', sep='\t', index_col=False)

y = data.pop('real')
X = data

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn import linear_model
model_lr1 = linear_model.LogisticRegression()
model_lr1.fit(X_train[['X1','X2','X3']], y_train)
lr1_y = model_lr1.predict(X_test[['X1','X2','X3']])
print(model_lr1.score(X_train[['X1','X2','X3']], y_train))

from sklearn import linear_model
model_lr2 = linear_model.LogisticRegression()
model_lr2.fit(X=X_train[['X4','X5','X6']], y=y_train)
lr2_y = model_lr2.predict(X_test[['X4','X5','X6']])
print(model_lr2.score(X_train[['X4','X5','X6']], y_train))

def proportions_diff_confint_ind(sample1, sample2, alpha = 0.05):    
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    
    p1 = float(sum(sample1)) / len(sample1)
    p2 = float(sum(sample2)) / len(sample2)
    
    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1)/ len(sample1) + p2 * (1 - p2)/ len(sample2))
    
    return (left_boundary, right_boundary)
    
def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    
    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2 
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))
    
def proportions_diff_z_test(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))
    
    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)
        
print("95%% confidence interval for a difference between proportions: [%f, %f]" %\
      proportions_diff_confint_ind(lr1_y, lr2_y))
      
print("p-value: %f" % proportions_diff_z_test(proportions_diff_z_stat_ind(lr1_y, lr2_y)))

print("p-value: %f" % proportions_diff_z_test(proportions_diff_z_stat_ind(lr1_y, lr2_y), 'less'))

def proportions_diff_confint_rel(sample1, sample2, alpha = 0.05):
    z = scipy.stats.norm.ppf(1 - alpha / 2.)
    sample = list(zip(sample1, sample2))
    n = len(sample1)
        
    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])
    
    left_boundary = float(f - g) / n  - z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    right_boundary = float(f - g) / n  + z * np.sqrt(float((f + g)) / n**2 - float((f - g)**2) / n**3)
    return (left_boundary, right_boundary)
    
def proportions_diff_z_stat_rel(sample1, sample2):
    sample = list(zip(sample1, sample2))
    n = len(sample1)
    
    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])
    
    return float(f - g) / np.sqrt(f + g - float((f - g)**2) / n )
    
print("95%% confidence interval for a difference between proportions: [%f, %f]" \
      % proportions_diff_confint_rel(lr1_y, lr2_y))
      
print("p-value: %f" % proportions_diff_z_test(proportions_diff_z_stat_rel(lr1_y, lr2_y)))

print("p-value: %f" % proportions_diff_z_test(proportions_diff_z_stat_rel(lr1_y, lr2_y), 'greater'))