# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 17:12:22 2016

@author: McSim
"""
import pandas as pd
import numpy as np
from sklearn import metrics, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score, train_test_split

df_train = pd.read_csv('logit_insurance.csv', sep = ',', index_col='INDEX')
df_test = pd.read_csv('logit_insurance_test.csv', sep = ',', index_col='INDEX')

y_train = pd.DataFrame(df_train.TARGET_FLAG)
y_test = pd.DataFrame(df_test.TARGET_FLAG)
y_train_add = pd.DataFrame(df_train.TARGET_AMT)
y_test_add = pd.DataFrame(df_test.TARGET_AMT)
X_train = df_train.drop(['TARGET_FLAG', 'TARGET_AMT'], axis = 1)
X_test = df_test.drop(['TARGET_FLAG', 'TARGET_AMT'], axis = 1)

X_train['INCOME'] = X_train['INCOME'].str.replace('$', '')
X_train['INCOME'] = X_train['INCOME'].str.replace(',', '')
X_train['INCOME'] = X_train['INCOME'].astype(np.float)
X_test['INCOME'] = X_test['INCOME'].str.replace('$', '')
X_test['INCOME'] = X_test['INCOME'].str.replace(',', '')
X_test['INCOME'] = X_test['INCOME'].astype(np.float)

X_train['BLUEBOOK'] = X_train['BLUEBOOK'].str.replace('$', '')
X_train['BLUEBOOK'] = X_train['BLUEBOOK'].str.replace(',', '')
X_train['BLUEBOOK'] = X_train['BLUEBOOK'].astype(np.float)
X_test['BLUEBOOK'] = X_test['BLUEBOOK'].str.replace('$', '')
X_test['BLUEBOOK'] = X_test['BLUEBOOK'].str.replace(',', '')
X_test['BLUEBOOK'] = X_test['BLUEBOOK'].astype(np.float)

X_train['HOME_VAL'] = X_train['HOME_VAL'].str.replace('$', '')
X_train['HOME_VAL'] = X_train['HOME_VAL'].str.replace(',', '')
X_train['HOME_VAL'] = X_train['HOME_VAL'].astype(np.float)
X_test['HOME_VAL'] = X_test['HOME_VAL'].str.replace('$', '')
X_test['HOME_VAL'] = X_test['HOME_VAL'].str.replace(',', '')
X_test['HOME_VAL'] = X_test['HOME_VAL'].astype(np.float)

X_train['OLDCLAIM'] = X_train['OLDCLAIM'].str.replace('$', '')
X_train['OLDCLAIM'] = X_train['OLDCLAIM'].str.replace(',', '')
X_train['OLDCLAIM'] = X_train['OLDCLAIM'].astype(np.float)
X_test['OLDCLAIM'] = X_test['OLDCLAIM'].str.replace('$', '')
X_test['OLDCLAIM'] = X_test['OLDCLAIM'].str.replace(',', '')
X_test['OLDCLAIM'] = X_test['OLDCLAIM'].astype(np.float)

categorical_f = ['PARENT1', 'EDUCATION', 'MSTATUS', 'SEX', 'JOB', 'CAR_USE', 'CAR_TYPE',
                 'RED_CAR', 'CLM_FREQ', 'REVOKED', 'URBANICITY']
numerical_f = ['KIDSDRIV', 'AGE', 'HOMEKIDS', 'YOJ',
       'INCOME', 'HOME_VAL', 'TRAVTIME', 'BLUEBOOK', 'TIF',
       'OLDCLAIM', 'MVR_PTS', 'CAR_AGE']

X_train['PARENT1'] = X_train['PARENT1'].str.replace('No', '1')
X_train['PARENT1'] = X_train['PARENT1'].str.replace('Yes', '2')
X_train['PARENT1'] = X_train['PARENT1'].astype(np.int)
X_test['PARENT1'] = X_test['PARENT1'].str.replace('No', '1')
X_test['PARENT1'] = X_test['PARENT1'].str.replace('Yes', '2')
X_test['PARENT1'] = X_test['PARENT1'].astype(np.int)

X_train['EDUCATION'] = X_train['EDUCATION'].str.replace('<High School', '1')
X_train['EDUCATION'] = X_train['EDUCATION'].str.replace('z_High School', '2')
X_train['EDUCATION'] = X_train['EDUCATION'].str.replace('Bachelors', '3')
X_train['EDUCATION'] = X_train['EDUCATION'].str.replace('Masters', '4')
X_train['EDUCATION'] = X_train['EDUCATION'].str.replace('PhD', '5')
X_train['EDUCATION'] = X_train['EDUCATION'].astype(np.int)
X_test['EDUCATION'] = X_test['EDUCATION'].str.replace('<High School', '1')
X_test['EDUCATION'] = X_test['EDUCATION'].str.replace('z_High School', '2')
X_test['EDUCATION'] = X_test['EDUCATION'].str.replace('Bachelors', '3')
X_test['EDUCATION'] = X_test['EDUCATION'].str.replace('Masters', '4')
X_test['EDUCATION'] = X_test['EDUCATION'].str.replace('PhD', '5')
X_test['EDUCATION'] = X_test['EDUCATION'].astype(np.int)

X_train['MSTATUS'] = X_train['MSTATUS'].str.replace('z_No', '1')
X_train['MSTATUS'] = X_train['MSTATUS'].str.replace('Yes', '2')
X_train['MSTATUS'] = X_train['MSTATUS'].astype(np.int)
X_test['MSTATUS'] = X_test['MSTATUS'].str.replace('z_No', '1')
X_test['MSTATUS'] = X_test['MSTATUS'].str.replace('Yes', '2')
X_test['MSTATUS'] = X_test['MSTATUS'].astype(np.int)

X_train['SEX'] = X_train['SEX'].str.replace('z_F', '1')
X_train['SEX'] = X_train['SEX'].str.replace('M', '2')
X_train['SEX'] = X_train['SEX'].astype(np.int)
X_test['SEX'] = X_test['SEX'].str.replace('z_F', '1')
X_test['SEX'] = X_test['SEX'].str.replace('M', '2')
X_test['SEX'] = X_test['SEX'].astype(np.int)

X_train['JOB'] = X_train['JOB'].str.replace('Student', '1')
X_train['JOB'] = X_train['JOB'].str.replace('Home Maker', '2')
X_train['JOB'] = X_train['JOB'].str.replace('Clerical', '3')
X_train['JOB'] = X_train['JOB'].str.replace('z_Blue Collar', '4')
X_train['JOB'] = X_train['JOB'].str.replace('Manager', '5')
X_train['JOB'] = X_train['JOB'].str.replace('Lawyer', '6')
X_train['JOB'] = X_train['JOB'].str.replace('Doctor', '7')
X_train['JOB'] = X_train['JOB'].str.replace('Professional', '8')
X_train['JOB'].fillna(0, inplace=True)
X_train['JOB_NAN'] = np.where(X_train['JOB'] == 0, 1, 0)
X_train['JOB'] = X_train['JOB'].astype(np.int)
X_test['JOB'] = X_test['JOB'].str.replace('Student', '1')
X_test['JOB'] = X_test['JOB'].str.replace('Home Maker', '2')
X_test['JOB'] = X_test['JOB'].str.replace('Clerical', '3')
X_test['JOB'] = X_test['JOB'].str.replace('z_Blue Collar', '4')
X_test['JOB'] = X_test['JOB'].str.replace('Manager', '5')
X_test['JOB'] = X_test['JOB'].str.replace('Lawyer', '6')
X_test['JOB'] = X_test['JOB'].str.replace('Doctor', '7')
X_test['JOB'] = X_test['JOB'].str.replace('Professional', '8')
X_test['JOB'].fillna(0, inplace=True)
X_test['JOB_NAN'] = np.where(X_test['JOB'] == 0, 1, 0)
X_test['JOB'] = X_test['JOB'].astype(np.int)

categorical_f = ['PARENT1', 'EDUCATION', 'MSTATUS', 'SEX', 'JOB', 'JOB_NAN', 'CAR_USE', 'CAR_TYPE',
                 'RED_CAR', 'CLM_FREQ', 'REVOKED', 'URBANICITY'] # Добавим новый признак в список
                 
X_train['CAR_USE'] = X_train['CAR_USE'].str.replace('Commercial', '1')
X_train['CAR_USE'] = X_train['CAR_USE'].str.replace('Private', '2')
X_train['CAR_USE'] = X_train['CAR_USE'].astype(np.int)
X_test['CAR_USE'] = X_test['CAR_USE'].str.replace('Commercial', '1')
X_test['CAR_USE'] = X_test['CAR_USE'].str.replace('Private', '2')
X_test['CAR_USE'] = X_test['CAR_USE'].astype(np.int)

X_train['CAR_TYPE'] = X_train['CAR_TYPE'].str.replace('z_SUV', '1')
X_train['CAR_TYPE'] = X_train['CAR_TYPE'].str.replace('Minivan', '2')
X_train['CAR_TYPE'] = X_train['CAR_TYPE'].str.replace('Pickup', '3')
X_train['CAR_TYPE'] = X_train['CAR_TYPE'].str.replace('Sports Car', '4')
X_train['CAR_TYPE'] = X_train['CAR_TYPE'].str.replace('Van', '5')
X_train['CAR_TYPE'] = X_train['CAR_TYPE'].str.replace('Panel Truck', '6')
X_train['CAR_TYPE'] = X_train['CAR_TYPE'].astype(np.int)

X_test['CAR_TYPE'] = X_test['CAR_TYPE'].str.replace('z_SUV', '1')
X_test['CAR_TYPE'] = X_test['CAR_TYPE'].str.replace('Minivan', '2')
X_test['CAR_TYPE'] = X_test['CAR_TYPE'].str.replace('Pickup', '3')
X_test['CAR_TYPE'] = X_test['CAR_TYPE'].str.replace('Sports Car', '4')
X_test['CAR_TYPE'] = X_test['CAR_TYPE'].str.replace('Van', '5')
X_test['CAR_TYPE'] = X_test['CAR_TYPE'].str.replace('Panel Truck', '6')
X_test['CAR_TYPE'] = X_test['CAR_TYPE'].astype(np.int)

X_train['RED_CAR'] = X_train['RED_CAR'].str.replace('no', '1')
X_train['RED_CAR'] = X_train['RED_CAR'].str.replace('yes', '2')
X_train['RED_CAR'] = X_train['RED_CAR'].astype(np.int)
X_test['RED_CAR'] = X_test['RED_CAR'].str.replace('no', '1')
X_test['RED_CAR'] = X_test['RED_CAR'].str.replace('yes', '2')
X_test['RED_CAR'] = X_test['RED_CAR'].astype(np.int)

X_train['REVOKED'] = X_train['REVOKED'].str.replace('No', '1')
X_train['REVOKED'] = X_train['REVOKED'].str.replace('Yes', '2')
X_train['REVOKED'] = X_train['REVOKED'].astype(np.int)
X_test['REVOKED'] = X_test['REVOKED'].str.replace('No', '1')
X_test['REVOKED'] = X_test['REVOKED'].str.replace('Yes', '2')
X_test['REVOKED'] = X_test['REVOKED'].astype(np.int)

X_train['URBANICITY'] = X_train['URBANICITY'].str.replace('Highly Urban/ Urban', '1')
X_train['URBANICITY'] = X_train['URBANICITY'].str.replace('z_Highly Rural/ Rural', '2')
X_train['URBANICITY'] = X_train['URBANICITY'].astype(np.int)
X_test['URBANICITY'] = X_test['URBANICITY'].str.replace('Highly Urban/ Urban', '1')
X_test['URBANICITY'] = X_test['URBANICITY'].str.replace('z_Highly Rural/ Rural', '2')
X_test['URBANICITY'] = X_test['URBANICITY'].astype(np.int)

X_train['YOJ'].fillna(100, inplace=True)
X_train['YOJ_NAN'] = np.where(X_train['YOJ'] == 100, 1, 0)
X_test['YOJ'].fillna(100, inplace=True)
X_test['YOJ_NAN'] = np.where(X_test['YOJ'] == 100, 1, 0)

X_train['INCOME'].fillna(-1, inplace=True)
X_train['INCOME_NAN'] = np.where(X_train['INCOME'] == -1, 1, 0)
X_test['INCOME'].fillna(-1, inplace=True)
X_test['INCOME_NAN'] = np.where(X_test['INCOME'] == -1, 1, 0)

X_train['AGE'].fillna(0, inplace=True)
X_train['AGE_NAN'] = np.where(X_train['AGE'] == 0, 1, 0)
X_test['AGE'].fillna(0, inplace=True)
X_test['AGE_NAN'] = np.where(X_test['AGE'] == 0, 1, 0)

X_train['HOME_VAL'].fillna(-1, inplace=True)
X_train['HOME_VAL_NAN'] = np.where(X_train['HOME_VAL'] == -1, 1, 0)
X_test['HOME_VAL'].fillna(-1, inplace=True)
X_test['HOME_VAL_NAN'] = np.where(X_test['HOME_VAL'] == -1, 1, 0)

X_train['CAR_AGE'].fillna(-3, inplace=True)
X_train['CAR_AGE_NAN'] = np.where(X_train['CAR_AGE'] == -3, 1, 0)
X_test['CAR_AGE'].fillna(-3, inplace=True)
X_test['CAR_AGE_NAN'] = np.where(X_test['CAR_AGE'] == -3, 1, 0)

categorical_f = ['PARENT1', 'EDUCATION', 'MSTATUS', 'SEX', 'JOB', 'JOB_NAN', 'CAR_USE', 'CAR_TYPE',
                 'RED_CAR', 'CLM_FREQ', 'REVOKED', 'URBANICITY', 'JOB_NAN', 'YOJ_NAN', 'INCOME_NAN',
                 'AGE_NAN', 'HOME_VAL_NAN', 'CAR_AGE_NAN' ] # Добавим новые признаки в список

X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size = 0.3, random_state=1)

print('==== Решающее дерево ====') #####################################################
# fit a CART model to the data
model_dtc = DecisionTreeClassifier(random_state=241)
model_dtc.fit(X1, y1)
#print(model)
## make predictions
#expected = target
#predicted = model.predict(source)
## summarize the fit of the model
#print(metrics.classification_report(expected, predicted))
#print(metrics.confusion_matrix(expected, predicted))
print('Важности признаков:', model_dtc.feature_importances_)
print('Точность модели:', model_dtc.score(X2, y2))

print('==== Логистическая регрессия ====') #####################################################
model_lr1 = linear_model.LogisticRegression(random_state=241)
model_lr1.fit(X1, y1)
Y1 = y1.reset_index()
Y1 = Y1.drop(['INDEX'], axis = 1)
Y2 = y2.reset_index()
Y2 = Y2.drop(['INDEX'], axis = 1)
print('Точность модели:', model_lr1.score(X2, Y2))

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

print('==== AdaBoostClassifier ====') #####################################################
model_adb = AdaBoostClassifier(random_state=241)
model_adb.fit(X1, y1)
print('Важности признаков:', model_adb.feature_importances_)
print('Точность модели:', model_adb.score(X2, y2))

print('==== BaggingClassifier ====') #####################################################
model_bgc = BaggingClassifier(random_state=241)
model_bgc.fit(X1, y1)
#print('Важности признаков:', model_bgc.feature_importances_)
print('Точность модели:', model_bgc.score(X2, y2))

print('==== ExtraTreesClassifier ====') #####################################################
model_etc = ExtraTreesClassifier(random_state=241)
model_etc.fit(X1, y1)
print('Важности признаков:', model_etc.feature_importances_)
print('Точность модели:', model_etc.score(X2, Y2))

print('==== GradientBoostingClassifier ====') #####################################################
model_gdb = GradientBoostingClassifier(random_state=241)
model_gdb.fit(X1, Y1)
print('Важности признаков:', model_gdb.feature_importances_)
print('Точность модели:', model_gdb.score(X2, Y2))

print('==== RandomForestClassifier ====') #####################################################
model_rfc = RandomForestClassifier(random_state=241)
model_rfc.fit(X1, y1)
print('Важности признаков:', model_rfc.feature_importances_)
print('Точность модели:', model_rfc.score(X2, Y2))

#################### Регуляризация градиентного бустинга ####################

#import matplotlib.pyplot as plt
#labels = y1['TARGET_FLAG'].unique()
#original_params = {'n_estimators': 500, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
#                   'min_samples_split': 5}
#plt.figure(figsize=[15,15])
#for label, color, setting in [('No shrinkage', 'orange',
#                               {'learning_rate': 1.0, 'subsample': 1.0}),
#                              ('learning_rate=0.1', 'turquoise',
#                               {'learning_rate': 0.1, 'subsample': 1.0}),
#                              ('subsample=0.5', 'blue',
#                               {'learning_rate': 1.0, 'subsample': 0.5}),
#                              ('learning_rate=0.1, subsample=0.5', 'gray',
#                               {'learning_rate': 0.1, 'subsample': 0.5}),
#                              ('learning_rate=0.1, max_features=2', 'magenta',
#                               {'learning_rate': 0.1, 'max_features': 2})]:
#    params = dict(original_params)
#    params.update(setting)
#    clf = GradientBoostingClassifier(**params)
#    clf.fit(X1, y1)
#
#    y = y2.as_matrix(columns=['TARGET_FLAG'])
#
#    # compute test set deviance
#    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
#
#    for i, y_pred in enumerate(list(clf.staged_decision_function(X2))):
#        # clf.loss_ assumes that y_test[i] in {0, 1}
#        test_deviance[i] = clf.loss_(y, y_pred)
#
#    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
#            '-', color=color, label=label)
#
#plt.legend(loc='upper left')
#plt.xlabel('Boosting Iterations')
#plt.ylabel('Test Set Deviance')
#
#plt.show()

############ Ищем оптимаьлнео количество итерраций бустинга ###################
#
#from sklearn.cross_validation import KFold
#
#X_1 = X1.reset_index()
#X_2 = X2.reset_index()
#
## Generate data (adapted from G. Ridgeway's gbm example)
#n_samples = 100
#
## Fit classifier with out-of-bag estimates
#params = {'n_estimators': 120, 'max_depth': 3, 'subsample': 0.5,
#          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
#clf = GradientBoostingClassifier(**params)
#
#clf.fit(X_1, Y1)
#acc = clf.score(X_2, Y2)
#print("Accuracy: {:.4f}".format(acc))
#
#n_estimators = params['n_estimators']
#x = np.arange(n_estimators) + 1
#
#
#def heldout_score(clf, X_test, y_test):
#    """compute deviance scores on ``X_test`` and ``y_test``. """
#    score = np.zeros((n_estimators,), dtype=np.float64)
#    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
#        score[i] = clf.loss_(y_test, y_pred)
#    return score
#
#
#def cv_estimate(n_folds=3):
#    cv = KFold(n=X_1.shape[0], n_folds=n_folds)
#    cv_clf = GradientBoostingClassifier(**params)
#    val_scores = np.zeros((n_estimators,), dtype=np.float64)
#    for train, test in list(cv):
#        cv_clf.fit(X_1[train], Y1[train])
#        val_scores += heldout_score(cv_clf, X_1[test], Y1[test])
#    val_scores /= n_folds
#    return val_scores
#
#
## Estimate best n_estimator using cross-validation
#cv_score = cv_estimate(3)
#
## Compute best n_estimator for test data
#test_score = heldout_score(clf, X_2, Y2)
#
## negative cumulative sum of oob improvements
#cumsum = -np.cumsum(clf.oob_improvement_)
#
## min loss according to OOB
#oob_best_iter = x[np.argmin(cumsum)]
#
## min loss according to test (normalize such that first loss is 0)
#test_score -= test_score[0]
#test_best_iter = x[np.argmin(test_score)]
#
## min loss according to cv (normalize such that first loss is 0)
#cv_score -= cv_score[0]
#cv_best_iter = x[np.argmin(cv_score)]
#
## color brew for the three curves
#oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
#test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
#cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))
#
## plot curves and vertical lines for best iterations
#plt.plot(x, cumsum, label='OOB loss', color=oob_color)
#plt.plot(x, test_score, label='Test loss', color=test_color)
#plt.plot(x, cv_score, label='CV loss', color=cv_color)
#plt.axvline(x=oob_best_iter, color=oob_color)
#plt.axvline(x=test_best_iter, color=test_color)
#plt.axvline(x=cv_best_iter, color=cv_color)
#
## add three vertical lines to xticks
#xticks = plt.xticks()
#xticks_pos = np.array(xticks[0].tolist() +
#                      [oob_best_iter, cv_best_iter, test_best_iter])
#xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +
#                        ['OOB', 'CV', 'Test'])
#ind = np.argsort(xticks_pos)
#xticks_pos = xticks_pos[ind]
#xticks_label = xticks_label[ind]
#plt.xticks(xticks_pos, xticks_label)
#
#plt.legend(loc='upper right')
#plt.ylabel('normalized loss')
#plt.xlabel('number of iterations')
#
#plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn import preprocessing

X_scaled = preprocessing.scale(X1)
Y_1 = y1.as_matrix(columns=['TARGET_FLAG'])
Y_2 = y2.as_matrix(columns=['TARGET_FLAG'])

kf = KFold(n=len(X_scaled), n_folds=5, shuffle=True, random_state=42)

kMeans = list()
KNC_scores = []
for i in range(1):
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    neigh.fit(X_scaled, Y_1)
    #KNeighborsClassifier(...)
    #print(neigh.predict(X))
    #print(str(i+1) + " : " + str(neigh.score(X_scaled, y1)))
    array = cross_val_score(estimator=neigh, X=X_scaled, y=Y_1, cv=kf, scoring='accuracy')
    KNC_scores.append([i+1,neigh.score(X_scaled, y1), array]) 
    m = array.mean()
    print("Accuracy: %0.2f (+/- %0.2f)" % (m, array.std() * 2))
    kMeans.append(m)