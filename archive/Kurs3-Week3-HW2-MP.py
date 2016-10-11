# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 10:40:43 2016

@author: McSim
"""

import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn

data = pandas.read_csv("train.csv", na_values="NaN")

data.head()

real_features = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
                 "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5"]
discrete_features = ["Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32"]
cat_features = data.columns.drop(real_features).drop(discrete_features).drop(["Id", "Response"]).tolist()

data[real_features].describe()

data[discrete_features].describe()

data.shape

# Код 1. Постройте гистограммы.
df = pandas.DataFrame(data[real_features])
df.hist(bins=100, figsize=(20, 20))

df = pandas.DataFrame(data[discrete_features])
df.hist(bins=100, figsize=(10, 10))

# Зря сделанная работа
#print("Есть ли среди признаков такие, которые появляются только 1 раз?")
#print("Среди целочисленных признаков:")
#print([1 in [[pandas.value_counts(data[x].values) for x in discrete_features][[y][0]].values for y in range(len(discrete_features))][z] for z in range(len(discrete_features))])
#print("\nСреди вещественных признаков:")
#print([1 in [[pandas.value_counts(data[x].values) for x in real_features][[y][0]].values for y in range(len(real_features))][z] for z in range(len(real_features))])

print("Есть ли среди признаков константные?")
print("Среди целочисленных признаков:")
print([len([pandas.Series(data[x].values.ravel()).unique() for x in discrete_features][y]) == 1 for y in range(len(discrete_features))])
print("\nСреди вещественных признаков:")
print([len([pandas.Series(data[x].values.ravel()).unique() for x in real_features][y]) == 1 for y in range(len(real_features))])
print("\nСреди категориальных признаков:")
print([len([pandas.Series(data[x].values.ravel()).unique() for x in cat_features][y]) == 1 for y in range(len(cat_features))])

seaborn.pairplot(data[real_features+["Response"]].drop(
        ["Employment_Info_4", "Employment_Info_6", "Insurance_History_5", "Product_Info_4"], axis=1), 
        hue="Response", diag_kind="kde")
        
# Код 2. Постройте pairplot для целочисленных признаков
seaborn.pairplot(data[discrete_features+["Response"]], hue="Response", diag_kind="kde")

seaborn.heatmap(data[real_features].corr(), square=True)

print(data[real_features].corr()>0.9)

fig, axes = plt.subplots(11, 10, figsize=(20, 20), sharey=True)
for i in range(len(cat_features)):
    seaborn.countplot(x=cat_features[i], data=data, ax=axes[i / 10, i % 10])

print('Есть ли константные признаки среди категориальных:')
print(True in [len([pandas.Series(data[x].values.ravel()).unique() for x in cat_features][y]) == 1 for y in range(len(cat_features))])
print('Есть ли признаки с количеством возможных категорий (число значений признака) больше 5?')
print(True in [len([pandas.Series(data[x].values.ravel()).unique() for x in cat_features][y]) > 5 for y in range(len(cat_features))])

# Код 3. Постройте countplot
fig, axes = plt.subplots(3, 8, figsize=(20, 30), sharey=True)
for i in range(3):
    for j in range(8):
        seaborn.countplot(x=['Medical_Keyword_23', 'Medical_Keyword_39', 'Medical_Keyword_45'][i], data=data[(data['Response']==j+1)], ax=axes[i, j])
        
seaborn.countplot(data.Response)

from sklearn.utils import shuffle
from sklearn.preprocessing import scale

sdata = shuffle(data, random_state=321)
# del data   # удалите неперемешанные данные, если не хватает оперативной памяти

subset_l  = 1000
selected_features = real_features[:-4]
objects_with_nan = sdata.index[np.any(np.isnan(sdata[selected_features].values), axis=1)]   
data_subset = scale(sdata[selected_features].drop(objects_with_nan, axis=0)[:subset_l])
response_subset = sdata["Response"].drop(objects_with_nan, axis=0)[:subset_l]

from sklearn.manifold import TSNE
import matplotlib.cm as cm # импортируем цветовые схемы, чтобы рисовать графики.

# Код 4. Присвойте переменной tsne_representation результат понижения размерности методом tSNE с параметрами по умолчанию
tsne = TSNE(n_components = 2, random_state = 321)
tsne_representation = tsne.fit_transform(data_subset)

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(data.Response), colors):
    plt.scatter(tsne_representation[response_subset.values==y, 0], 
                tsne_representation[response_subset.values==y, 1], c=c, alpha=0.5, label=str(y))
plt.legend()
plt.show()

from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances

# Код 5. Присвойте переменной MDS_transformed результат понижения размерности методом MDS с параметрами по умолчанию
# Не забудьте зафиксировать random_state=321
mds = MDS(n_components = 2, random_state = 321)
MDS_transformed = mds.fit_transform(data_subset)

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(response_subset), colors):
    plt.scatter(MDS_transformed[response_subset.values==y, 0], 
                MDS_transformed[response_subset.values==y, 1], 
                c=c, alpha=0.5, label=str(y))
plt.legend()
#plt.xlim(-5, 5)   # масса точек концентрируется в этом масштабе
#plt.ylim(-5, 5)   # рекомендуем сначала отобразить визуализацию целиком, а затем раскомментировать эти строки.
plt.show()

# Код 6. Присвойте переменной MDS_transformed_cos результат понижения размерности методом MDS с косинусной метрикой
pair_dist = pairwise_distances(X=data_subset)
mds = MDS(n_components = 2, random_state=321, dissimilarity="precomputed")
MDS_transformed_cos = mds.fit_transform(pair_dist)

colors = cm.rainbow(np.linspace(0, 1, len(set(response_subset))))
for y, c in zip(set(response_subset), colors):
    plt.scatter(MDS_transformed_cos[response_subset.values[:subset_l]==y, 0], 
                MDS_transformed_cos[response_subset.values[:subset_l]==y, 1], 
                c=c, alpha=0.5, label=str(y))
plt.legend()
plt.show()

from sklearn import svm
person_features = ["Ins_Age", "Ht", "Wt", "BMI"]
svm_ = svm.OneClassSVM(gamma=10, nu=0.01) 
svm_.fit(sdata[person_features])
labels = svm_.predict(sdata[person_features])
print((labels==1).mean())

# Код 7. Постройте 6 графиков
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
colors = cm.rainbow(np.linspace(0, 1, len(set(person_features))))
for y, c in zip(set(person_features), colors):
    plt.scatter(MDS_transformed[person_features.values==y, 0], 
                MDS_transformed[person_features.values==y, 1], 
                c=c, alpha=0.5, label=str(y))
