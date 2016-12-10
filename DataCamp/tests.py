# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:36:09 2016

@author: McSim
"""

# Create a string str1
str1 = "Introduction with strings"

# Now store the length of string str1 in variable str_len 
str_len = len(str1)

str_new = "Machine Learning is awesome!"
# Print last eight characters of string str_new (the length of str_new is 28 characters).
print(str_new[len(str_new)-8:])

str2 = "I am doing a course Introduction to Hackathon using "
str3 = "Python"

# Write a line of code to store concatenated string of str2 and str3 into variable str4
str4 = str2 + str3

# import library pandas
import pandas as pd

# Import training data as train
train = pd.read_csv("https://s3-ap-southeast-1.amazonaws.com/av-datahack-datacamp/train.csv")

# Import testing data as test
test = pd.read_csv("https://s3-ap-southeast-1.amazonaws.com/av-datahack-datacamp/test.csv")

# Print top 5 observation of train dataset
print(train.head(5))

# Store total number of observation in training dataset
train_length = len(train)

# Store total number of columns in testing data set
test_col = len(test.columns)

#Training and Testing data set are loaded in train and test dataframe respectively

# Look at the summary of numerical variables for train data set
df= train.describe()
print (df)

# Print the unique values and their frequency of variable Property_Area
df1=train.Property_Area.value_counts()

# Training and Testing dataset are loaded in train and test dataframe respectively
# Plot histogram for variable LoanAmount
train.LoanAmount.hist(bins=50)

# Plot a box plot for variable LoanAmount by variable Gender of training data set
train.boxplot(column='LoanAmount', by = 'Gender')

# Training and Testing dataset are loaded in train and test dataframe respectively

# Approved Loan in absolute numbers
loan_approval = train['Loan_Status'].value_counts()['Y']

# Two-way comparison: Credit History and Loan Status
twowaytable = pd.crosstab(train["Credit_History"], train["Loan_Status"], margins=True)

# How many missing values in variable "Self_Employed" ?
n_missing_value_Self_Employed = train['Self_Employed'].isnull().sum()

# Variable Loan amount has missing values or not?
LoanAmount_have_missing_value = train['LoanAmount'].isnull().sum() > 0

# Check variables have missing values in test data set
number_missing_values_test_data = test.isnull().sum()

# Impute missing value of LoanAmount with 168 for test data set
test['LoanAmount'].fillna(168, inplace=True)

# Impute missing value of Gender (Male is more frequent category)
train['Gender'].fillna('Male',inplace=True)


# Impute missing value of Credit_History ( 1 is more frequent category)
train['Credit_History'].fillna(1,inplace=True)

# Training and Testing datasets are loaded in variable train and test dataframe respectively

# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']

import numpy as np
# Perform log transformation of TotalIncome to make it closer to normal
train['TotalIncome_log']= np.log(train['TotalIncome'])

#import module for label encoding
from sklearn.preprocessing import LabelEncoder

#train and test dataset is already loaded in the enviornment
# Perform label encoding for variable 'Married'
number = LabelEncoder()
train['Married_new'] = number.fit_transform(train['Married'].astype(str))

train_modified = train.copy()
train_modified['Education'] = number.fit_transform(train_modified['Education'].astype(str))
train_modified['Gender'] = number.fit_transform(train_modified['Gender'].astype(str))

# Import linear model of sklearn
import sklearn.linear_model

# Create object of Logistic Regression
model=sklearn.linear_model.LogisticRegression()

#train_modified and test_modified already loaded in the workspace
#Import module for Logistic regression
import sklearn.linear_model

# Select three predictors Credit_History, Education and Gender
predictors =['Credit_History','Education','Gender']

# Converting predictors and outcome to numpy array
x_train = train_modified[predictors].values
y_train = train_modified['Loan_Status'].values

# Model Building
model = sklearn.linear_model.LogisticRegression()
model.fit(x_train, y_train)

#test_modified already loaded in the workspace
test_modified = test.copy()
test_modified['Education'] = number.fit_transform(test_modified['Education'].astype(str))
test_modified['Gender'] = number.fit_transform(test_modified['Gender'].astype(str))

test_modified['Loan_Amount_Term'].fillna(test_modified['Loan_Amount_Term'].mean(), inplace=True)
test_modified['Credit_History'].fillna(1, inplace=True)

# Select three predictors Credit_History, Education and Gender
predictors =['Credit_History','Education','Gender']

# Converting predictors and outcome to numpy array
x_test = test_modified[predictors].values

#Predict Output
predicted= model.predict(x_test)

#Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)

#Store it to test dataset
test_modified['Loan_Status']=predicted

#Output file to make submission
test_modified.to_csv("Submission1.csv",columns=['Loan_ID','Loan_Status'])

# Import tree module of sklearn
import sklearn.tree

# Create object of DecisionTreeClassifier
model = sklearn.tree.DecisionTreeClassifier()

#train_modified and test_modified already loaded in the workspace
#Import module for Decision tree
import sklearn.tree

# Select three predictors Credit_History, Education and Gender
predictors =['Credit_History','Education','Gender']

# Converting predictors and outcome to numpy array
x_train = train_modified[predictors].values
y_train = train_modified['Loan_Status'].values

# Model Building
model = sklearn.tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

# Converting predictors and outcome to numpy array
x_test = test_modified[predictors].values

#Predict Output
predicted= model.predict(x_test)

#Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)

#Store it to test dataset
test_modified['Loan_Status']=predicted

#Output file to make submission
test_modified.to_csv("Submission1.csv",columns=['Loan_ID','Loan_Status'])

# Import ensemble module from sklearn
import sklearn.ensemble

# Create object of RandomForestClassifier
model=sklearn.ensemble.RandomForestClassifier()

#train_modified and test_modified already loaded in the workspace
#Import module for Random Forest
import sklearn.ensemble

# Select three predictors Credit_History, Education and Gender
predictors =['Credit_History','Education','Gender']

# Converting predictors and outcome to numpy array
x_train = train_modified[predictors].values
y_train = train_modified['Loan_Status'].values

# Model Building
model = sklearn.ensemble.RandomForestClassifier()
model.fit(x_train, y_train)

# Converting predictors and outcome to numpy array
x_test = test_modified[predictors].values

#Predict Output
predicted= model.predict(x_test)

#Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)

#Store it to test dataset
test_modified['Loan_Status']=predicted

#Output file to make submission
test_modified.to_csv("Submission1.csv",columns=['Loan_ID','Loan_Status'])

predictors=['ApplicantIncome', 'CoapplicantIncome', 'Credit_History','Dependents', 'Education', 'Gender', 'LoanAmount',
            'Loan_Amount_Term', 'Married', 'Property_Area', 'Self_Employed', 'TotalIncome','Log_TotalIncome']

featimp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)

print (featimp)

