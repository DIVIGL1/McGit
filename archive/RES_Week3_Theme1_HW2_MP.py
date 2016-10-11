# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 00:39:47 2016

@author: McSim
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

newsgroups = datasets.fetch_20newsgroups(subset='all', categories = ['alt.atheism', 'sci.space'])

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X, y)
idf = vectorizer.idf_
print(dict(zip(vectorizer.get_feature_names(), idf)))

#grid = {'C' : np.power(10.0, np.arange(-5, 6))}
#cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
#clf = SVC(kernel='linear', random_state=241)
#gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#gs.fit(X_tfidf, y)

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=241, shrinking=True,
  tol=0.001, verbose=False)
#SVC(C=1, kernel='linear', random_state=241)
clf.fit(X_tfidf, y)
#y_pred = clf.predict(X_test)
#score_perceptron = metrics.accuracy_score(y_test, y_pred)
words = vectorizer.get_feature_names()
coefs = clf.coef_.toarray()[0]

coefs = list(map(lambda x: abs(x), coefs))

ten = heapq.nlargest(10, coefs)
ten_words = []

for coef in ten:
    for i in range(0, len(coefs)):
        if coefs[i] == coef:
            ten_words.append(words[i])

ten_words.sort()
print(ten_words)

#file = open('ans.txt', 'w')
#print(' '.join(ten_words), file=file, sep='', end='')
#file.close()
