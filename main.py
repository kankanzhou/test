# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


df=pd.read_csv("~/Desktop/HDB/hdb.csv")

#print(df)
print(df.columns)
X = df[['floor_area_sqm','lease_commence_date','year','month','town','flat_type'
        ,'storey_range','street_name','flat_model']]
y = df['resale_price']

X=pd.get_dummies(X, prefix=['town', 'flat_type','storey_range','street_name','flat_model'])

#print(X)

#X = df[['floor_area_sqm']]
#y=np.log(y)
y = y.values.ravel()

#print(X,y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Create linear regression object
lr = linear_model.LinearRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = lr.predict(X_test)


# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
print("LR RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, y_pred)))
# Explained variance score: 1 is perfect prediction
print('LR R2: %.2f' % r2_score(y_test, y_pred))

rf = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=10)

# Train the model using the training sets
rf.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = rf.predict(X_test)
y_pred_t = rf.predict(X_train)

# Explained variance score: 1 is perfect prediction
print('RF R2: %.2f' % r2_score(y_test, y_pred))
print('RF R2(train): %.2f' % r2_score(y_train, y_pred_t))


from sklearn.model_selection import cross_val_predict

predicted = cross_val_predict(lr, X, y, cv=10)
print("LR-CV RMSE: %.2f"
      % sqrt(mean_squared_error(y, predicted)))
print('LR-CV R2: %.2f' % r2_score(y, predicted))



predicted = cross_val_predict(rf, X, y, cv=10)
print("RF-CV RMSE: %.2f"
      % sqrt(mean_squared_error(y, predicted)))
print('RF-CV R2: %.2f' % r2_score(y, predicted))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print(importances,indices)














