
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\14th\MLR\House_data.csv')

dataset.head()

print(dataset.isnull().any())

print(dataset.dtypes)

dataset = dataset.drop(['id','date'],axis = 1)

with sns.plotting_context('notebook',font_scale=2.5):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']],
        hue = 'bedrooms', palette = 'tab20',size = 6)
g.set(xticklabels=[])

X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =1/3, random_state =0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

slope = regressor.coef_
slope

constant = regressor.intercept_
constant

bais = regressor.score(X_train, y_train)
bais

variance = regressor.score(X_test, y_test)
variance
# Backward Elimination

import statsmodels.formula.api as sm
import statsmodels.api as sm

def backwardElimination(x,SL):
    numVars = len(x[0])
    temp = np.zeros((21613,19)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar>SL:
            for j in range(0, numVars -i):
                if(regressor_OLS.pvalues[j].astype(float) ==maxVar):
                    temp[:,j] = x[:,j]
                    x = np.delete(x,j,1)
                    tmp_regressor = sm.OLS(y,x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if(adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j,1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_Modeled = backwardElimination(X_opt, SL)
    
    
    
    
    
    
    
    
    
    
    