# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 16:47:30 2016

@author: McSim
"""

import numpy as np
import pandas as pd
from sklearn import feature_extraction
from sklearn import linear_model
import re
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

# 1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах
# из файла salary-train.csv
df = pd.read_csv("salary-train.csv", index_col = False, delimiter=",")

# 2. Проведите предобработку:

# 2.1 Приведите тексты к нижнему регистру (text.lower()).
df['FullDescription'] = df['FullDescription'].str.lower()

# 2.2 Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова. 
#     Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text). 
#     Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:
#     train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
df['FullDescription'] = df['FullDescription'].str.replace('[^a-zA-Z0-9]',' ', regex = True)

# 2.3 Примените TfidfVectorizer для преобразования текстов в векторы признаков. 
#     Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer).
vectoriser = feature_extraction.text.TfidfVectorizer(min_df=5)
vectors = list(vectoriser.fit_transform(df['FullDescription']).toarray())
print(np.max(vectors))

# 2.4 Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. 
df['LocationNormalized'].fillna('nan', inplace=True)
df['ContractTime'].fillna('nan', inplace=True)

# 2.5 Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
enc = DictVectorizer()
X_train_categ = enc.fit_transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(df[['LocationNormalized', 'ContractTime']].to_dict('records'))

# 2.6 Объедините все полученные признаки в одну матрицу "объекты-признаки". 
#     Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными. 
#     Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
matrix = hstack([vectors,X_train_categ])
print(matrix)

