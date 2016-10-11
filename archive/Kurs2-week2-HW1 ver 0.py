# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 00:36:16 2016

@author: McSim
"""

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score as cv_score

# ЗАДАНИЕ 1

def plot_scores(d_scores):
    n_components = np.arange(1,len(d_scores)+1)
    plt.plot(n_components, d_scores, 'b', label='PCA scores')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('cv scores')
    plt.legend(loc='lower right')
    plt.show()
    
def write_answer_1(optimal_d):
    with open("pca_answer1.txt", "w") as fout:
        fout.write(str(optimal_d))
        
data = pd.read_csv('data_task1.csv')

# place your code here
max_score = 0 # Объявим переменную для максимального среднего логарифма правдоподобия данных моделей
optimal_d = 0 # Объявим переменную для количества компонент при максимальном среднем логарифме правдоподобия данных
d_scores = list() # В этот список будем собирать все полученные средние логарифмы правдоподобия данных

for d in range(1,data.shape[1]+1):
    model = PCA(n_components=d)
    scores = cv_score(model, data)
    d_scores.append(scores.mean())
    if scores.mean() > max_score or d == 1:
        max_score = scores.mean()
        optimal_d = d

print('Cредний логарифм правдоподобия по всем моделям = {0}'.format(sum(d_scores)/d))
print('Максимальный логарифм правдоподобия равне {0} и достигается при d равном {1}'.format(max_score, optimal_d))

plot_scores(d_scores)

#write_answer_1(optimal_d)

# ЗАДАНИЕ 2

def plot_variances(d_variances):
    n_components = np.arange(1,d_variances.size+1)
    plt.plot(n_components, d_variances, 'b', label='Component variances')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('variance')
    plt.legend(loc='upper right')
    plt.show()
    
def write_answer_2(optimal_d):
    with open("pca_answer2.txt", "w") as fout:
        fout.write(str(optimal_d))
        
data = pd.read_csv('data_task2.csv')

model = PCA(n_components = data.shape[1])
model.fit(data)
components = model.components_
d_variances = model.explained_variance_
transformed_data = model.transform(data)

plot_variances(d_variances)

delta_var = dict()
for i in range(1,len(d_variances)):
    delta_var[i] = d_variances[i]-d_variances[i-1]

sorted_delta_var = sorted(delta_var.items(), key=lambda x: x[1] , reverse=True)

print(r'Эффективная размерность данных $\hat{d}$ = ' + str(sorted_delta_var[-1][0]))

#write_answer_2(sorted_delta_var[-1][0])

# ЗАДАНИЕ 3

from sklearn import datasets

def plot_iris(transformed_data, target, target_names):
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(transformed_data[target == i, 0],
                    transformed_data[target == i, 1], c=c, label=target_name)
    plt.legend()
    plt.show()
    
def write_answer_3(list_pc1, list_pc2):
    with open("pca_answer3.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc1]))
        fout.write(" ")
        fout.write(" ".join([str(num) for num in list_pc2]))

# загрузим датасет iris
iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

# place your code here
mean_data = data.mean(axis=0)
new_data = data - mean_data
model = PCA(n_components = 4)
model.fit(new_data, target)
components = model.components_
d_variances = model.explained_variance_
transformed_data = model.transform(new_data)
mean_transformed_data = transformed_data.mean(axis=0)
transformed_data = transformed_data - mean_transformed_data

plot_iris(transformed_data, target, target_names)

r = np.ndarray([4,2], dtype=float)
for j in range(4):
    for k in range(2):
        chislitel = 0
        znamenatel_1 = 0
        znamenatel_2 = 0
        for i in range(150):
            chislitel += (new_data[i,j] - new_data.mean(axis=0)[j]) * (transformed_data[i,k] - transformed_data.mean(axis=0)[k])
            znamenatel_1 += (new_data[i,j] - new_data.mean(axis=0)[j])**2
            znamenatel_2 += (transformed_data[i,k] - transformed_data.mean(axis=0)[k])**2
        r[j,k] = chislitel / (znamenatel_1 * znamenatel_2)**0.5

list_pc1 = list([1,3,4])
list_pc2 = list([2])
write_answer_3(list_pc1, list_pc2)

# ЗАДАНИЕ 4

from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import RandomizedPCA

def write_answer_4(list_pc):
    with open("pca_answer4.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in list_pc]))

data = fetch_olivetti_faces(shuffle=True, random_state=0).data
image_shape = (64, 64)

new_data = data # - data.mean(axis=0)
model = RandomizedPCA(n_components = data.shape[0])
model.fit(new_data)
components = model.components_
d_variances = model.explained_variance_
transformed_data = model.transform(new_data)
new_transformed_data = transformed_data #- transformed_data.mean(axis=0)

num_main_components = 10

r = np.ndarray([data.shape[1],num_main_components], dtype=float)
for j in range(data.shape[1]):
    print(j)
    for k in range(num_main_components):
        chislitel = 0
        znamenatel_1 = 0
        znamenatel_2 = 0
        chislitel = (new_data[:,j] - new_data.mean(axis=0)[j]) * (transformed_data[:,k] - transformed_data.mean(axis=0)[k])
        znamenatel_1 = (new_data[:,j] - new_data.mean(axis=0)[j])**2
        znamenatel_2 = (transformed_data[:,k] - transformed_data.mean(axis=0)[k])**2
        r[j,k] = chislitel.sum() / (znamenatel_1.sum() * znamenatel_2.sum())**0.5

q = new_data.dot(r)
s = np.ndarray([q.shape[0], q.shape[1]], dtype=float)
f_mean = q.mean(axis=0)
t = np.ndarray([q.shape[1]], dtype=float)
for k in range(q.shape[1]):
    for i in range(q.shape[0]):
        t[k] += (q[i,k] - f_mean[k])**2
        
for i in range(q.shape[0]):
    for k in range(q.shape[1]):
        s[i,k] = (q[i,k] - f_mean[k])**2 / t[k]

list_pc = s.argmax(axis=0)

for k in list_pc:
    print('Лицо ' + str(k))
    plt.imshow(data[k,:].reshape(image_shape))
    plt.show()

write_answer_4(list_pc)

## ПРИМЕРЫ
#C1 = np.array([[10,0],[0,0.5]])
#phi = np.pi/3
#C2 = np.dot(C1, np.array([[np.cos(phi), np.sin(phi)],
#                          [-np.sin(phi),np.cos(phi)]]))
#
#data = np.vstack([np.random.multivariate_normal(mu, C1, size=50),
#                  np.random.multivariate_normal(mu, C2, size=50)])
#plt.scatter(data[:,0], data[:,1])
## построим истинные интересующие нас компоненты
#plt.plot(data[:,0], np.zeros(data[:,0].size), color="g")
#plt.plot(data[:,0], 3**0.5*data[:,0], color="g")
## обучим модель pca и построим главные компоненты
#model = PCA(n_components=2)
#model.fit(data)
#plot_principal_components(data, model, scatter=False, legend=False)
#c_patch = mpatches.Patch(color='c', label='Principal components')
#plt.legend(handles=[g_patch, c_patch])
#plt.draw()
#
#C = np.array([[0.5,0],[0,10]])
#mu1 = np.array([-2,0])
#mu2 = np.array([2,0])
#
#data = np.vstack([np.random.multivariate_normal(mu1, C, size=50),
#                  np.random.multivariate_normal(mu2, C, size=50)])
#plt.scatter(data[:,0], data[:,1])
## обучим модель pca и построим главные компоненты
#model = PCA(n_components=2)
#model.fit(data)
#plot_principal_components(data, model)
#plt.draw()

