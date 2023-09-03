import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\14th\MLR\Investment.csv')

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]

X =pd.get_dummies(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2, random_state =0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)              

slope = regressor.coef_
slope

constant = regressor.intercept_
constant

bias = regressor.score(X_train, y_train)
bias  # 95%

variance = regressor.score(X_test, y_test)
variance # 93%
# we have bias and variance are 95% and 93% so this is the good model

#***** we build the model so far

import statsmodels.formula.api as sm

value = 42467

X = np.append(arr = np.full((50,1),value,dtype=int),values=X,axis=1)
# X = np.append(arr= np.ones((50,1)).astype(int),values=X, axxis =1)
import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5]]

# OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm
X_opt  = X[:,[0,1,2,3]]

regressor_OLS = sm.OLS(endog =y, exog =X_opt).fit()
regressor_OLS.summary()

X_opt  = X[:,[0,1,3]]

regressor_OLS = sm.OLS(endog =y, exog =X_opt).fit()
regressor_OLS.summary()

X_opt  = X[:,[0,1]]

regressor_OLS = sm.OLS(endog =y, exog =X_opt).fit()
regressor_OLS.summary()
