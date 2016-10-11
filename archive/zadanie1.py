# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import numpy as np
import pandas as pn
dt = pn.read_csv('C:\\Users\\McSim\\Documents\\Python Scripts\\titanic.csv', index_col='PassengerId')
print('Survived column values:')
print(dt.Survived.value_counts())
print()
dt0=pn.DataFrame(dt,columns=['Sex'])
dt1=dt[dt['Survived']==1]
dt2=dt[dt['Survived']==0]
dt1c=dt1['Survived'].count()
dt0c=dt2['Survived'].count()
dts = dt1c/(dt1c+dt0c)*100
print('Survivals % = ',dts)
print()
dtam=dt['Age'].median()
print('Age median = ',dtam)
dtaa=dt['Age'].mean()
print('Age average = ',dtaa)
print()
dtcorr=dt.corr(method='pearson', min_periods=1)
print('Correlation = ',dtcorr)
print()
dt['Names']=dt['Name'].str.extract('(\. )([A-Za-z]*)()',expand=True)[1]
dt['OrNames']=dt['Name'].str.extract('(\()([A-Za-z]*)()',expand=True)[1]
dtw=dt[dt['Sex']=='female']
#writer = pn.ExcelWriter('test.xlsx', engine='xlsxwriter')
#dt.to_excel(writer, sheet_name='Sheet1')
#writer = pn.ExcelWriter('C:\\Users\\McSim\\Documents\\Python Scripts\\test.xlsx', engine='xlsxwriter')
#dt.to_excel(writer, sheet_name='Sheet1')
#writer.save()
print('Женские имена (после обращения(Мисс, Миссис,..):)',dtw.Names.value_counts()[:5])
