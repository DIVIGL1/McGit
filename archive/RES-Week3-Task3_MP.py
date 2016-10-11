# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:06:04 2016

@author: McSim
"""

# Метрики качества классификации

import pandas as pd
import numpy as np
from sklearn import metrics

df_class = pd.read_csv('classification.csv', index_col=None)

TP = df_class.groupby(['true']).get_group(1).pred.sum()
FN = df_class.groupby(['true']).get_group(1).shape[0] - TP
FP = df_class.groupby(['true']).get_group(0).pred.sum()
TN = df_class.groupby(['true']).get_group(0).shape[0] - FP

matrix = [[TP,FP],[FN,TN]]

file = open('ans1.txt', 'w')
print((str(TP)+' '+str(FP)+' '+str(FN)+' '+str(TN)), file=file, sep='', end='')
file.close()

df_metrics_accuracy = metrics.accuracy_score(df_class.true, df_class.pred)
df_metrics_precision = metrics.precision_score(df_class.true, df_class.pred)
df_metrics_recall = metrics.recall_score(df_class.true, df_class.pred)
df_metrics_f1 = metrics.f1_score(df_class.true, df_class.pred)

file = open('ans2.txt', 'w')
print((str(round(df_metrics_accuracy,2))+' '+str(round(df_metrics_precision,2))+' '+str(round(df_metrics_recall,2))+' '+str(round(df_metrics_f1,2))), file=file, sep='', end='')
file.close()

df_scores = pd.read_csv('scores.csv', index_col=None)

roc_log = metrics.roc_auc_score(df_scores.true, df_scores.score_logreg)
roc_svm = metrics.roc_auc_score(df_scores.true, df_scores.score_svm)
roc_knn = metrics.roc_auc_score(df_scores.true, df_scores.score_knn)
roc_tree = metrics.roc_auc_score(df_scores.true, df_scores.score_tree)

roc_log.max()
roc_svm.max()
roc_knn.max()
roc_tree.max()

file = open('ans3.txt', 'w')
print("score_logreg", file=file, sep='', end='')
file.close()

precision, recall, thresholds = metrics.precision_recall_curve(df_scores.true, df_scores.score_logreg)
print(round(precision[:86].max(),2))

precision, recall, thresholds = metrics.precision_recall_curve(df_scores.true, df_scores.score_svm)
print(round(precision[:89].max(),2))

precision, recall, thresholds = metrics.precision_recall_curve(df_scores.true, df_scores.score_knn)
print(round(precision[:41].max(),2))

precision, recall, thresholds = metrics.precision_recall_curve(df_scores.true, df_scores.score_tree)
print(round(precision[:6].max(),2))

file = open('ans4.txt', 'w')
print("score_tree", file=file, sep='', end='')
file.close()
