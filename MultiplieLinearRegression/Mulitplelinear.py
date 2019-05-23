#multilinear regression
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset
dataset=pd.read_csv("C:\\Users\\HP\\Desktop\\machine learning geeks\\mystudy\\P14-Multiple-Linear-Regression\\Multiple_Linear_Regression\\50_Startups.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
X


#Avoiding the dummy variable trap
X=X[:,1:]

#splitting the data into trainning and testing
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#No need of doing features scaling in mlr because library will take care of it
#fitting 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test results
y_pred=regressor.predict(X_test)


#Bulding the optimal model using backward elimination
import statsmodels.formula.api as sm

