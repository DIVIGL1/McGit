# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 18:50:10 2016

@author: McSim
"""
import pandas as pd
import numpy as np

df = pd.read_csv('credit_card_default_analysis.csv', sep = ',', index_col='ID')