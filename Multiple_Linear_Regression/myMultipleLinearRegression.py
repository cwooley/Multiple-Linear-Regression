# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 11:54:34 2017

@author: Charles
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
             
# Avoiding the Dummy Variable Trap
# Always need to omit one dummy variable... because math
X = X[:, 1:]             

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


## Feature Scaling is not necessary here.
# Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Creating the model and fitting to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set Results
y_pred = regressor.predict(X_test)

# Building a model for Backward Elimination
import statsmodels.formula.api as sm

# Add column of 1's as a coeficcient for b0
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#Making the optimal matrix of features with signifigant p values for dependant variables
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Fit ordinary least squares to Xopt and y
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#Remove column @ index 2 because it has the highest P value
X_opt = X[:, [0, 1, 3, 4, 5]]
# Fit ordinary least squares to Xopt and y
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#Remove column @ index 1 because it has the highest P value
X_opt = X[:, [0, 3, 4, 5]]
# Fit ordinary least squares to Xopt and y
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#Remove column @ index 4 because it has the highest P value
X_opt = X[:, [0, 3, 5]]
# Fit ordinary least squares to Xopt and y
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()

#Remove column @ index 4 because it has the highest P value Just barely didn't make the cut
X_opt = X[:, [0, 3]]
# Fit ordinary least squares to Xopt and y
regressor_OLS = sm.OLS(endog = y, exog= X_opt).fit()
regressor_OLS.summary()


                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            