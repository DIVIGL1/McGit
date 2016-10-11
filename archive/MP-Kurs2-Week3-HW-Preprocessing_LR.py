# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 17:58:04 2016

@author: McSim
"""

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')

data = pd.read_csv('data.csv')
print(data.shape)

X = data.drop('Grant.Status', 1)
y = data['Grant.Status']

numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3', 
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))

def calculate_means(numeric_data):
    means = np.zeros(numeric_data.shape[1])
    for j in range(numeric_data.shape[1]):
        to_sum = numeric_data.iloc[:,j]
        indices = np.nonzero(~numeric_data.iloc[:,j].isnull())[0]
        correction = np.amax(to_sum[indices])
        to_sum /= correction
        for i in indices:
            means[j] += to_sum[i]
        means[j] /= indices.size
        means[j] *= correction
    return pd.Series(means, numeric_data.columns)

# place your code here
# Функция calculate_means() масштабирует элементы полученной ею на входе таблицы, поэтому создаём для неё отдельную таблицу

X_real_zeros = X.drop(categorical_cols, axis = 1)     # Создаём клон датафрэйма Х без категориальных признаков (столбцов).
if X_real_zeros.isnull().any().any():                 # Проверяем на наличие NaN, чтобы не делать лишних вычислений, если не надо
    X_real_zeros[np.isnan(X_real_zeros)] = 0          # Заменяем все NaN на 0. Метод np.nan_to_num() преобразовывает датафрейм в numpy.ndarray
print("Есть ли в датафрейме с вещественными признаками значения типа NaN?", X_real_zeros.isnull().any().any()) # Удостоверимся, что значений NaN в новом датафрейме нет.

X_real_mean = X.drop(categorical_cols, axis = 1)
inds = np.where(np.isnan(X_real_mean))
# Функция calculate_means() масштабирует элементы полученной ею на входе таблицы
num_col_means = calculate_means(X.drop(categorical_cols, axis = 1)) 
for i in range(len(inds[0])):
    X_real_mean[numeric_cols[inds[1][i]]][inds[0][i]] = num_col_means[inds[1][i]]
print("Есть ли во втором датафрейме с вещественными признаками значения типа NaN?", X_real_mean.isnull().any().any())

X_cat = X.drop(numeric_cols, axis = 1)
X_cat.fillna("NA", inplace=True)
X_cat = X_cat.astype(str)
print("Есть ли во втором датафрейме с категориальными признаками значения типа NaN?", X_cat.isnull().any().any())

from sklearn.feature_extraction import DictVectorizer as DV
encoder = DV(sparse = False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())

from sklearn.cross_validation import train_test_split

(X_train_real_zeros, 
 X_test_real_zeros, 
 y_train, y_test) = train_test_split(X_real_zeros, y, 
                                     test_size=0.3, 
                                     random_state=0)
(X_train_real_mean, 
 X_test_real_mean) = train_test_split(X_real_mean, 
                                      test_size=0.3, 
                                      random_state=0)
(X_train_cat_oh,
 X_test_cat_oh) = train_test_split(X_cat_oh, 
                                   test_size=0.3, 
                                   random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score

def plot_scores(optimizer):
    scores = [[item[0]['C'], 
               item[1], 
               (np.sum((item[2]-item[1])**2)/(item[2].size-1))**0.5] for item in optimizer.grid_scores_]
    scores = np.array(scores)
    plt.semilogx(scores[:,0], scores[:,1])
    plt.fill_between(scores[:,0], scores[:,1]-scores[:,2], 
                                  scores[:,1]+scores[:,2], alpha=0.3)
    plt.show()
    
def write_answer_1(auc_1, auc_2):
    auc = (auc_1 + auc_2)/2
    with open("preprocessing_lr_answer1.txt", "w") as fout:
        fout.write(str(auc))
        
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'random_state':[0]}
cv = 3
penalty="l2" # Значение по умолчанию. Стоит ли указывать явным образом?
n_jobs = 1 # Пробовать варианты?

X_train_with_zeros = np.hstack((X_train_real_zeros,X_train_cat_oh))
X_train_with_means = np.hstack((X_train_real_mean,X_train_cat_oh))
X_test_with_zeros = np.hstack((X_test_real_zeros,X_test_cat_oh))
X_test_with_means = np.hstack((X_test_real_mean,X_test_cat_oh))

estimator = LogisticRegression(penalty)

optimizer = GridSearchCV(estimator, param_grid, cv = cv, scoring = 'accuracy', n_jobs = n_jobs)
# Сначала для данных с нулями
optimizer.fit(X_train_with_zeros, y_train)
#print(optimizer.predict(X_test_with_zeros))
#print(optimizer.predict_proba(X_test_with_zeros))
print(optimizer.best_estimator_)
print(optimizer.best_score_)
print(optimizer.best_params_)
test_predictions_with_zeros = optimizer.predict(X_test_with_zeros)
probability_predictions_zeros = optimizer.predict_proba(X_test_with_zeros)
plot_scores(optimizer)

optimizer.fit(X_train_with_means, y_train)
print(optimizer.best_estimator_)
print(optimizer.best_score_)
print(optimizer.best_params_)
test_predictions_with_means = optimizer.predict(X_test_with_means)
probability_predictions_means = optimizer.predict_proba(X_test_with_means)
plot_scores(optimizer)

from sklearn import metrics

auc_1 = roc_auc_score(y_test, test_predictions_with_means) 
print("Метрика качества AUC ROC для данных со средними =", auc_1)
print(roc_auc_score(y_test, probability_predictions_means[:,1]))

auc_2 = metrics.roc_auc_score(y_test, test_predictions_with_zeros)
print("Метрика качества AUC ROC для данных с нулями =", auc_2)
print(roc_auc_score(y_test, probability_predictions_zeros[:,1]))

#write_answer_1(auc_1, auc_2)

from pandas.tools.plotting import scatter_matrix

data_numeric = pd.DataFrame(X_train_real_zeros, columns=numeric_cols)
list_cols = ['Number.of.Successful.Grant.1', 'SEO.Percentage.2', 'Year.of.Birth.1']
scatter_matrix(data_numeric[list_cols], alpha=0.5, figsize=(10, 10))
plt.show()

from sklearn.preprocessing import StandardScaler

# place your code here
scaler = StandardScaler()
X_train_real_scaled = scaler.fit_transform(X_train_real_zeros)
X_test_real_scaled = scaler.transform(X_test_real_zeros)

data_numeric_scaled = pd.DataFrame(X_train_real_scaled, columns=numeric_cols)
list_cols = ['Number.of.Successful.Grant.1', 'SEO.Percentage.2', 'Year.of.Birth.1']
pd.scatter_matrix(data_numeric_scaled[list_cols], alpha=0.5, figsize=(10, 10))
plt.show()

def write_answer_2(auc):
    with open("preprocessing_lr_answer2.txt", "w") as fout:
        fout.write(str(auc))
        
# place your code here
X_train_with_zeros = np.hstack((X_train_real_scaled,X_train_cat_oh))
X_test_with_zeros = np.hstack((X_test_real_scaled,X_test_cat_oh))
optimizer.fit(X_train_with_zeros, y_train)
print(optimizer.best_estimator_)
print(optimizer.best_score_)
print(optimizer.best_params_)
test_predictions_with_zeros = optimizer.best_estimator_.predict(X_test_with_zeros)
plot_scores(optimizer)
auc = metrics.roc_auc_score(y_test, test_predictions_with_zeros)
print("Метрика качества AUC ROC для отмасштабированных данных с нулями =", auc)
#write_answer_2(auc)
