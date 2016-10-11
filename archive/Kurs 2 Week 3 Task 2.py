# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 13:16:22 2016

@author: McSim
"""

from sklearn import cross_validation, grid_search, linear_model, metrics
import pylab
import numpy as np
import pandas as pd

raw_data = pd.read_csv('bike_sharing_demand.csv', header = 0, sep = ',')

print(raw_data.head(), "\n")

print(raw_data.shape, "\n")

print(raw_data.isnull().values.any(), "\n")

print(raw_data.info(), "\n")

raw_data.datetime = raw_data.datetime.apply(pd.to_datetime)

raw_data['month'] = raw_data.datetime.apply(lambda x : x.month)
raw_data['hour'] = raw_data.datetime.apply(lambda x : x.hour)

print(raw_data.head(), "\n")

train_data = raw_data.iloc[:-1000, :]
hold_out_test_data = raw_data.iloc[-1000:, :]

print(raw_data.shape, train_data.shape, hold_out_test_data.shape, "\n")

print('train period from {} to {}'.format(train_data.datetime.min(), train_data.datetime.max()))
print('evaluation period from {} to {}'.format(hold_out_test_data.datetime.min(), hold_out_test_data.datetime.max()))

#обучение
train_labels = train_data['count'].values
train_data = train_data.drop(['datetime', 'count'], axis = 1)

#тест
test_labels = hold_out_test_data['count'].values
test_data = hold_out_test_data.drop(['datetime', 'count'], axis = 1)

pylab.figure(figsize = (16, 6))

pylab.subplot(1,2,1)
pylab.hist(train_labels)
pylab.title('train data')

pylab.subplot(1,2,2)
pylab.hist(test_labels)
pylab.title('test data')

pylab.show()

numeric_columns = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'month', 'hour']

train_data = train_data[numeric_columns]
test_data = test_data[numeric_columns]

print(train_data.head(), "\n")
print(test_data.head(), "\n")

regressor = linear_model.SGDRegressor(random_state = 0)

regressor.fit(train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, regressor.predict(test_data)), "\n")

print(test_labels[:10])

print(regressor.predict(test_data)[:10], "\n")

print(regressor.coef_, "\n")

from sklearn.preprocessing import StandardScaler

#создаем стандартный scaler
scaler = StandardScaler()
scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

regressor.fit(scaled_train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data)), "\n")

print(test_labels[:10], "\n")

print(regressor.predict(scaled_test_data)[:10], "\n")

print(regressor.coef_, "\n")

list(map(lambda x : '{:.2f}'.format(x), regressor.coef_))

print(train_data.head(), "\n")

print(train_labels[:10], "\n")

np.all(train_data.registered + train_data.casual == train_labels)

train_data.drop(['casual', 'registered'], axis = 1, inplace = True)
test_data.drop(['casual', 'registered'], axis = 1, inplace = True)

scaler.fit(train_data, train_labels)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

regressor.fit(scaled_train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, regressor.predict(scaled_test_data)), "\n")

print(map(lambda x : round(x, 2), regressor.coef_), "\n")
list(map(lambda x : '{:.2f}'.format(x), regressor.coef_))

from sklearn.pipeline import Pipeline

#создаем pipeline из двух шагов: scaling и классификация
pipeline = Pipeline(steps = [('scaling', scaler), ('regression', regressor)])

pipeline.fit(train_data, train_labels)
print(metrics.mean_absolute_error(test_labels, pipeline.predict(test_data), "\n")

print(pipeline.get_params().keys())

parameters_grid = {
    'regression__loss' : ['huber', 'epsilon_insensitive', 'squared_loss', ],
    'regression__n_iter' : [3, 5, 10, 50], 
    'regression__penalty' : ['l1', 'l2', 'none'],
    'regression__alpha' : [0.0001, 0.01],
    'scaling__with_mean' : [0., 0.5],
}

grid_cv = grid_search.GridSearchCV(pipeline, parameters_grid, scoring = 'mean_absolute_error', cv = 4)

grid_cv.fit(train_data, train_labels)

print(grid_cv.best_score_)
print(grid_cv.best_params_)

print(metrics.mean_absolute_error(test_labels, grid_cv.best_estimator_.predict(test_data)), "\n")

print(np.mean(test_labels), "\n")

test_predictions = grid_cv.best_estimator_.predict(test_data)
print(test_labels[:10])
print(test_predictions[:10])

pylab.figure(figsize=(16, 6))

pylab.subplot(1,2,1)
pylab.grid(True)
pylab.scatter(train_labels, pipeline.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, pipeline.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('no parameters setting')
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)

pylab.subplot(1,2,2)
pylab.grid(True)
pylab.scatter(train_labels, grid_cv.best_estimator_.predict(train_data), alpha=0.5, color = 'red')
pylab.scatter(test_labels, grid_cv.best_estimator_.predict(test_data), alpha=0.5, color = 'blue')
pylab.title('grid search')
pylab.xlim(-100,1100)
pylab.ylim(-100,1100)