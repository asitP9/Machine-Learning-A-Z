# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('50_Startups.csv')

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 4].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

onehotencoder=OneHotEncoder(categories='auto')
column_transformer=ColumnTransformer([('one_hot_encoder', onehotencoder, [3])], remainder='passthrough')
X=column_transformer.fit_transform(X)

X=X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, Y_train) 

y_pred=regressor.predict(X_test)

import statsmodels.api as sm
X=np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)


X_opt=np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
columnsInX_Opt=np.array([0, 1, 2, 3, 4, 5])
xOpt_Final=False

while (xOpt_Final==False):
    regressor_ols=sm.OLS(endog=Y, exog=X_opt).fit()
    max=regressor_ols.pvalues[0]
    finalindex=0
    # xOpt_Final=True

    for index, value in np.ndenumerate(columnsInX_Opt):
        if regressor_ols.pvalues[index]>max:
            max=regressor_ols.pvalues[index]
            finalIndex=index
    
    if max>0.05:
        X_opt=np.delete(X_opt, np.s_[finalIndex], axis=1) 
        columnsInX_Opt=np.delete(columnsInX_Opt, [finalIndex])
    
    elif max<=0.05:
        xOpt_Final=True
        print(columnsInX_Opt)
        xOpt_Final=True
        # print(regressor_ols.pvalues[i])
