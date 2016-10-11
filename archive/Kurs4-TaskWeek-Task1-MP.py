# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 23:43:15 2016

@author: McSim
"""

import pandas as pd
import numpy as np
import scipy
from scipy import stats
import statsmodels.stats.multitest as smm

df = pd.read_csv("gene_high_throughput_sequencing.csv", sep=",", index_col=False) 

df_diag_means = df.groupby(df['Diagnosis'], axis=0)
df_diag_means = df_diag_means.mean()

Fc_matrix = pd.DataFrame(index=['normal_neoplasia','neoplasia_cancer'], columns=df_diag_means.columns)
C = df_diag_means[df_diag_means.index == 'normal']
T = df_diag_means[df_diag_means.index == 'early neoplasia']
for i in C.columns:
    if T[i]['early neoplasia'] > C[i]['normal']:
        Fc_matrix[i]['normal_neoplasia'] = T[i]['early neoplasia'] / C[i]['normal']
    else:
        Fc_matrix[i]['normal_neoplasia'] = - C[i]['normal'] / T[i]['early neoplasia']

C = df_diag_means[df_diag_means.index == 'early neoplasia']
T = df_diag_means[df_diag_means.index == 'cancer']
for i in C.columns:
    if T[i]['cancer'] > C[i]['early neoplasia']:
        Fc_matrix[i]['neoplasia_cancer'] = T[i]['cancer'] / C[i]['early neoplasia']
    else:
        Fc_matrix[i]['neoplasia_cancer'] = - C[i]['early neoplasia'] / T[i]['cancer']

df_normal = df.copy()
df_normal = df_normal.drop(['Patient_id'], axis=1)
df_normal = df_normal[df['Diagnosis']=='normal']
df_normal = df_normal.drop(['Diagnosis'], axis=1)
df_normal.head()

df_cancer = df.copy()
df_cancer = df_cancer.drop(['Patient_id'], axis=1)
df_cancer = df_cancer[df['Diagnosis']=='cancer']
df_cancer = df_cancer.drop(['Diagnosis'], axis=1)
df_cancer.head()

df_neoplasia = df.copy()
df_neoplasia = df_neoplasia.drop(['Patient_id'], axis=1)
df_neoplasia = df_neoplasia[df['Diagnosis']=='early neoplasia']
df_neoplasia = df_neoplasia.drop(['Diagnosis'], axis=1)
df_neoplasia.head()

#### Задание 1 ######

p_norm_neoplasia = scipy.stats.ttest_ind(df_normal, df_neoplasia, equal_var = False).pvalue
sum1 = sum(1 for i in p_norm_neoplasia if i < 0.05)

p_neoplasia_cancer = scipy.stats.ttest_ind(df_neoplasia, df_cancer, equal_var = False).pvalue
sum2 = sum(1 for i in p_neoplasia_cancer if i < 0.05)

def save_answer(sum, nr):
    with open("Kurs4-TaskWeek-Task1-ans"+str(nr)+".txt", "w") as fout:
        fout.write(str(sum))

save_answer(int(sum1),11)
save_answer(int(sum2),12)

#### Задание 2 ######

p_norm_neoplasia_new = smm.multipletests(p_norm_neoplasia, alpha = 0.025, method = 'holm')
p_neoplasia_cancer_new = smm.multipletests(p_neoplasia_cancer, alpha = 0.025, method = 'holm')

inxs_norm_neoplasia_new = [i for i, j in enumerate(p_norm_neoplasia_new[0]) if j == True]
inxs_neoplasia_cancer_new = [i for i, j in enumerate(p_neoplasia_cancer_new[0]) if j == True]

filtered_inxs_norm_neoplasia_new = [j for i, j in enumerate(Fc_matrix.columns) if abs(Fc_matrix.ix[0,j])>1.5 and i in inxs_norm_neoplasia_new]
filtered_inxs_neoplasia_cancer_new = [j for i, j in enumerate(Fc_matrix.columns) if abs(Fc_matrix.ix[1,j])>1.5 and i in inxs_neoplasia_cancer_new]

save_answer(len(filtered_inxs_norm_neoplasia_new), 21)
save_answer(len(filtered_inxs_neoplasia_cancer_new), 22)

#### Задание 3 ######

reject, p_norm_neoplasia_new2, a1, a2 = smm.multipletests(p_norm_neoplasia, 
                                            alpha = 0.025, 
                                            method = 'fdr_bh')
inxs_norm_neoplasia_new2 = [i for i, j in enumerate(reject) if j == True]
filtered_inxs_norm_neoplasia_new2 = [j for i, j in enumerate(Fc_matrix.columns) if abs(Fc_matrix.ix[0,j])>1.5 and i in inxs_norm_neoplasia_new2]

reject, p_neoplasia_cancer_new2, a1, a2 = smm.multipletests(p_neoplasia_cancer, 
                                            alpha = 0.025, 
                                            method = 'fdr_bh')
inxs_neoplasia_cancer_new2 = [i for i, j in enumerate(reject) if j == True]
filtered_inxs_neoplasia_cancer_new2 = [j for i, j in enumerate(Fc_matrix.columns) if abs(Fc_matrix.ix[1,j])>1.5 and i in inxs_neoplasia_cancer_new2]

save_answer(len(filtered_inxs_norm_neoplasia_new2), 31)
save_answer(len(filtered_inxs_neoplasia_cancer_new2), 32)
