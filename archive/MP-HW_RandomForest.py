# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:37:19 2016

@author: McSim
"""

from sklearn import ensemble, cross_validation, learning_curve, metrics, datasets, tree
import numpy as np
import pandas as pd
import pylab

def write_answer(num, input_data):
    with open("MP-HW-RandomForest-"+str(num)+".txt", "w") as fout:
        fout.write(str(input_data))

digits = datasets.load_digits()
print(list(digits.keys()))
digits_X = digits.data
digits_y = digits.target

classifier = tree.DecisionTreeClassifier()
digits_scoring = cross_validation.cross_val_score(classifier, digits_X, digits_y, scoring = 'accuracy', cv = 10)
print("Средняя оценка качества:", digits_scoring.mean())
write_answer(1, digits_scoring.mean())

trees = ensemble.BaggingClassifier(classifier, n_estimators=100)
digits_scoring = cross_validation.cross_val_score(trees, digits_X, digits_y, scoring = 'accuracy', cv = 10)
print("Средняя оценка качества при 100 деревьях:", digits_scoring.mean())
write_answer(2, digits_scoring.mean())

trees = ensemble.BaggingClassifier(classifier, n_estimators=100, max_features = np.sqrt(digits_X.shape[1])/digits_X.shape[1])
digits_scoring = cross_validation.cross_val_score(trees, digits_X, digits_y, scoring = 'accuracy', cv = 10)
print("Средняя оценка качества при 100 деревьях и ограничении количества признаков:", digits_scoring.mean())
write_answer(3, digits_scoring.mean())

classifier = tree.DecisionTreeClassifier(max_features='sqrt')
trees = ensemble.BaggingClassifier(classifier, n_estimators=100)
digits_scoring = cross_validation.cross_val_score(trees, digits_X, digits_y, scoring = 'accuracy', cv = 10)
print("Средняя оценка качества при 100 деревьев и ограничении кол-ва признаков кажой ветки:", digits_scoring.mean())
write_answer(4, digits_scoring.mean())

rf_classifier = ensemble.RandomForestClassifier(n_estimators = 100, random_state = 1, max_features='sqrt')
digits_scoring = cross_validation.cross_val_score(rf_classifier, digits_X, digits_y, scoring = 'accuracy', cv = 10)
print("Средняя оценка качества Случайноголеса при 100 деревьях и ограничении кол-ва признаков:", digits_scoring.mean())

print("\n Сравнение показателей при разных вводных: \n")

print("При разном количестве деревьев:")
for i in [5,10,15,100]:
    
    rf_classifier = ensemble.RandomForestClassifier(n_estimators = i, random_state = 1, max_features='sqrt')
    train_sizes, train_scores, test_scores = learning_curve.learning_curve(rf_classifier, digits_X, digits_y,
                                                                           cv=10, scoring='accuracy')
    print(train_sizes)
    print(train_scores.mean(axis = 1))
    print(test_scores.mean(axis = 1))
    pylab.grid(True)
    pylab.title("Number of trees = "+str(i))
    pylab.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
    pylab.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
    pylab.ylim((0.0, 1.05))
    pylab.legend(loc='lower right')
    pylab.show()

print("При разном количестве признаков:")
for i in [5,10,40,50]:
    
    rf_classifier = ensemble.RandomForestClassifier(n_estimators = 50, random_state = 1, max_features=i)
    train_sizes, train_scores, test_scores = learning_curve.learning_curve(rf_classifier, digits_X, digits_y,
                                                                           cv=10, scoring='accuracy')
    print(train_sizes)
    print(train_scores.mean(axis = 1))
    print(test_scores.mean(axis = 1))
    pylab.grid(True)
    pylab.title("Number of features = "+str(i))
    pylab.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
    pylab.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
    pylab.ylim((0.0, 1.05))
    pylab.legend(loc='lower right')
    pylab.show()

print("При разной глубине:")
for i in [5,6]:
    
    rf_classifier = ensemble.RandomForestClassifier(n_estimators = 100, random_state = 1, max_features='sqrt', max_depth = i)
    train_sizes, train_scores, test_scores = learning_curve.learning_curve(rf_classifier, digits_X, digits_y,
                                                                           cv=10, scoring='accuracy')
    print(train_sizes)
    print(train_scores.mean(axis = 1))
    print(test_scores.mean(axis = 1))
    pylab.grid(True)
    pylab.title("Depth = "+str(i))
    pylab.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')
    pylab.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')
    pylab.ylim((0.0, 1.05))
    pylab.legend(loc='lower right')
    pylab.show()


