# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:48:00 2019

@author: pitam
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Dataset
df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

#splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting the simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)


#visualizing the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.savefig('fig1.png')
plt.show()

#visualizing the testing set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.savefig('fig2.png')
plt.show()