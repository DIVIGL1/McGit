# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 20:05:35 2016

@author: McSim
"""

import numpy as np
import re
import scipy
#from scipy import spatial
#from scipy import linalg
#Открываем файл, записываем его в виде списка в переменную f_list
f = open("C:\\Users\\McSim\\Documents\\My Ebooks\\Машинное обучение\\МФТИ\\Курс 1. Математика и Python для анализа данных\\Week 2\\Линейная алгебра. Матрицы\\sentences.txt")
f_sens = f.readlines()
#f.close()
f_sens = [i.lower() for i in f_sens] # переводим все элементы списка в нижний регистр
f_sens = [i.strip() for i in f_sens] # очищаем строки от лишних переносов в конце
f_sens = [" "+i+" " for i in f_sens] # добавляем пробелы в начале и конце предложения
f_sens = [re.sub('[\.\,-]\s',' ',i) for i in f_sens] # 
f_sens = [re.sub('[/)(]',' ',i) for i in f_sens] # 


#Открываем файл снова, записываем его в виде стринга в переменную f_words
#f = open("C:\\Users\\McSim\\Documents\\My Ebooks\\Машинное обучение\\МФТИ\\Курс 1. Математика и Python для анализа данных\\Week 2\\Линейная алгебра. Матрицы\\sentences.txt")
f.seek(0)
f_string = f.read()
f.close()
f_string = f_string.lower() # переводим весь текст разом в нижний регистр

f_string = re.sub('[\.\,-]\s',' ',f_string)
f_string = re.sub('[/)(]',' ',f_string)
f_allwords = re.split("[ \n]", f_string) # [^a-z] разбиваем текст на слова и записываем результат в список f_allwords
f_allwords = [i for i in f_allwords if i!="" and i!="'"] # Убираем пустые элементы
# Хорошо бы убрать апострофы и однобуквенные слова, но в задании этого нет.  and len(i)>1

# создаём новый список f_words без дублей
f_words = []
for i in f_allwords:
    if i not in f_words:
        f_words.append(i)

# Создаём словарь из списка, чтобы слову соответствовал конкретный индекс
f_dict= {i:f_words[i] for i in range(len(f_words))}

# Создаём матрицу размера n * d, где n — число предложений. 
# Элемент с индексом (i, j) равен количеству вхождений j-го слова в i-е предложение.
def f_freq(c,r):
    line=f_sens[r]    
    return line.count(" "+f_dict[c]+" ")
f_matrix=np.matrix([[f_freq(c,r) for c in range(len(f_words))] for r in range(len(f_sens))])

# Создаём список из косинусных расстояний от первого предложения до всех остальных, второй элемент - номер строки
f_cosines=[[scipy.spatial.distance.cosine(f_matrix[0],f_matrix[i]),i] for i in range(len(f_sens))]
f_cosines.sort()
print(f_cosines[1:3]) # выводим данные двух ближайших предложений
print("По косинусному расстоянию от предложения:")
print(f_sens[f_cosines[0][1]])
print("ближайшее предложение:")
print(f_sens[f_cosines[1][1]])
print("следующее предложение:")
print(f_sens[f_cosines[2][1]])

        