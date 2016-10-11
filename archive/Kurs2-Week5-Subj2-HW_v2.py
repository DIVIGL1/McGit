# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:22:44 2016

@author: McSim
"""

from sklearn import datasets, cross_validation
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
import numpy as np

def write_answer(num, input_data):
    with open("MP-NB-ans-"+str(num)+".txt", "w") as fout:
        fout.write(str(input_data))

DG_dataset = datasets.load_digits()
BC_dataset = datasets.load_breast_cancer()

#test_data_quota = 0.25
#
#DG_data_size = int(DG_dataset.data.shape[0]*(1-test_data_quota))
#BC_data_size = int(BC_dataset.data.shape[0]*(1-test_data_quota))
#DG_data_train, DG_data_test, DG_target_train, DG_target_test = DG_dataset.data[:DG_data_size], DG_dataset.data[DG_data_size:], DG_dataset.target[:DG_data_size], DG_dataset.target[DG_data_size:]
#BC_data_train, BC_data_test, BC_target_train, BC_target_test = BC_dataset.data[:BC_data_size], BC_dataset.data[BC_data_size:], BC_dataset.target[:BC_data_size], BC_dataset.target[BC_data_size:]
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(boston.data, boston.target, test_size = test_data_quota, random_state=0)

classificators = (BernoulliNB(), MultinomialNB(), GaussianNB())

print('Analysis of the dataset \'breast_cancer\'')
BC_clfs_scores = list()
for classificator in classificators:
    BC_scoring = cross_validation.cross_val_score(classificator, BC_dataset.data, BC_dataset.target)
    BC_clfs_scores.append(BC_scoring.mean())
    print('Classificator: ' + str(classificator))
    print('Score: mean = ' + str(BC_scoring.mean()) + '\n')
print("Maximum score is "+str(max(BC_clfs_scores)) + '\n')
write_answer(1, max(BC_clfs_scores))

print('Analysis of the dataset \'digits\'')
DG_clfs_scores = list()
for classificator in classificators:
    DG_scoring = cross_validation.cross_val_score(classificator, DG_dataset.data, DG_dataset.target)
    DG_clfs_scores.append(DG_scoring.mean())
    print('Classificator: ' + str(classificator))
    print('Score: mean = ' + str(DG_scoring.mean()) + '\n')
print("Maximum score is "+str(max(DG_clfs_scores)) + '\n')
write_answer(2, max(DG_clfs_scores))



