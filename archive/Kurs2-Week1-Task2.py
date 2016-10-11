# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:42:49 2016

@author: McSim
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def write_answer_to_file(answer, filename):
    with open(filename, 'w') as f_out:
        f_out.write(str(round(answer, 3)))

adver_data = pd.read_csv('advertising.csv')

print(adver_data.head(5))

adver_data.info()

X = adver_data.values[:,1:4] # Ваш код здесь
y = adver_data.values[:,4] # Ваш код здесь

means, stds = np.mean(X, axis=0), np.std(X, axis=0) # Ваш код здесь

X = (X - means) / stds # Ваш код здесь

# Ваш код здесь
X = np.hstack((X, np.ones(len(X)).reshape(len(X),1)))

def mserror(y, y_pred):
    # Ваш код здесь
    return 1/len(X)*(np.linalg.norm(y - y_pred))**2
    
answer1 = mserror(y, np.median(y, axis=0)) # Ваш код здесь
print(answer1)
write_answer_to_file(answer1, '1.txt')

def normal_equation(X, y):
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))  # Ваш код здесь
    
norm_eq_weights = normal_equation(X, y)
print(norm_eq_weights)

answer2 = norm_eq_weights[3] # Ваш код здесь
print(answer2)
write_answer_to_file(answer2, '2.txt')

def linear_prediction(X, w):
    # Ваш код здесь
    return X.dot(w)
    
answer3 = 1/len(X)*(np.linalg.norm(y - linear_prediction(X, norm_eq_weights)))**2 # Ваш код здесь
print(answer3)
write_answer_to_file(answer3, '3.txt')

def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    grad0 = w.dot(X[train_ind]) - y[train_ind] # Ваш код здесь
    grad1 = w.dot(X[train_ind]) - y[train_ind] # Ваш код здесь
    grad2 = w.dot(X[train_ind]) - y[train_ind] # Ваш код здесь
    grad3 = w.dot(X[train_ind]) - y[train_ind] # Ваш код здесь
    return  w - 2 * eta / len (X) * X[train_ind].T * np.array([grad0, grad1, grad2, grad3])
 
    
def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом. 
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Будем порождать псевдослучайные числа 
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)
        
    # Основной цикл
    while weight_dist > min_weight_dist and len(errors) < max_iter:
        # порождаем псевдослучайный 
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])
        
        # Ваш код здесь
        errors += [1 / len(X) * (y[random_ind] - X[random_ind].dot(w))**2]  
        w1 = stochastic_gradient_step(X, y, w, random_ind, eta)
        weight_dist = np.linalg.norm(w - w1)
        if weight_dist <= min_weight_dist:
            print("Достигли минимального расстояния при шаге. Итеррация "+str(len(errors)))
        if len(errors) == max_iter:
            print("Достигли максимальной итеррации "+str(len(errors)))            
        w = w1              
        
    return w, errors

stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, np.zeros(4), eta=1e-2, max_iter=1e5,
                                min_weight_dist=1e-8, seed=42, verbose=False)

plt.plot(range(50), stoch_errors_by_iter[:50])
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.show()

plt.plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.show()

print("Вектор весов, к которому сошелся метод:")
print(stoch_grad_desc_weights)

print("Среднеквадратичная ошибка на последней итерации:")
print(stoch_errors_by_iter[-1])

answer4 = 1/len(X)*(np.linalg.norm(y - X.dot(stoch_grad_desc_weights)))**2 # Ваш код здесь
print("Среднеквадратичная ошибка прогноза значений Sales в виде линейной модели с весами, найденными с помощью градиентного спуска =\n" + str(answer4))
write_answer_to_file(answer4, '4.txt')