# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 22:08:51 2016

@author: McSim
"""

import pandas as pd
import numpy as np # Подключение библиотеки для работы с матрицами
import matplotlib.pyplot as plt # Подключение библиотеки для отображения графиков
import scipy.stats as sts # Подключение статистической библиотеки
a, b = 0.5, 0.5
beta_crv = sts.beta(a, b)
sample = beta_crv.rvs(1000)
x = np.linspace(0,1,1000)
cdf = beta_crv.cdf(x)
plt.plot(x, cdf, label='theoretical CDF')