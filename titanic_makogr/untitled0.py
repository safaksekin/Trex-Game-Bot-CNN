# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 19:34:27 2022

@author: safak
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import os
import cv2
from sklearn.svm import SVR
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

d=pd.read_csv("train.csv")

path=""
df=pd.read_csv("train.csv")
df=df.dropna()

df_y=df["Survived"]

df=df.drop(["Survived"],axis=1)
df=df.drop(["PassengerId"],axis=1)
df=df.drop(["Name"],axis=1)
df=df.drop(["Ticket"],axis=1)
df=df.drop(["Fare"],axis=1)
df=df.drop(["Cabin"],axis=1)
df=df.drop(["Embarked"],axis=1)

df=pd.get_dummies(df,columns=["Sex"],prefix=["Sex"])
df=df.drop(["Sex_male"],axis=1)

#splitting the datas
x_train,x_test,y_train,y_test=train_test_split(df,df_y,test_size=0.25,random_state=42)

# machine learning (Radial Support Vector Machines)
"""svr_model=SVR("rbf").fit(x_train,y_train)

svr_params={"C":[0.1,0.2,0.4,0.8,3,4,5,10,15,20,25,30,40,50]}
svr_cv_model=GridSearchCV(svr_model, svr_params,cv=10)
svr_cv_model.fit(x_train,y_train)

svr_tuned=SVR("rbf",C=pd.Series(svr_cv_model.best_params_)[0]).fit(x_train,y_train)"""

"""lm=LinearRegression()
model=lm.fit(x_train,y_train)"""

"""knn_params={"n_neighbors":np.arange(1,30,1)}
knn_model=KNeighborsRegressor()
knn_cv_model=GridSearchCV(knn_model, knn_params,cv=10)
knn_cv_model.fit(x_train,y_train)

knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(x_train,y_train)"""


"""mlp_model=MLPRegressor().fit(x_train,y_train)

mlp_params={"alpha":[0.1,0.01,0.02,0.03,0.2,0.002,0.003,0.005],
            "activation":["relu","logistic"]}

mlp_cv_model=GridSearchCV(mlp_model,mlp_params,cv=10)
mlp_cv_model.fit(x_train,y_train)

mlp_tuned=MLPRegressor(alpha=0.005,activation="relu")
mlp_tuned.fit(x_train,y_train)"""

# saving the model
"""model_name="titanic_svr.pickle"
pickle.dump(svr_tuned,open(model_name,"wb"))"""

"""model_name="titanic_lin.pickle"
pickle.dump(lm,open(model_name,"wb"))"""

"""model_name="titanic_knn.pickle"
pickle.dump(knn_tuned,open(model_name,"wb"))"""

"""model_name="titanic_ysa.pickle"
pickle.dump(mlp_tuned,open(model_name,"wb"))"""

my_model=pickle.load(open("titanic_ysa.pickle","rb"))

print("mean squared error score: {}".format(np.sqrt(-cross_val_score(my_model,x_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()))

# test the model
x_test.index=np.arange(0,len(x_test),1)
y_test.index=np.arange(0,len(y_test),1)

counter=0
for i in x_test.index:
    if round(my_model.predict(pd.DataFrame(x_test.iloc[i]).T)[0])==y_test.iloc[i]:
        counter+=1

print("Successful Prediction Rate: {}".format((counter/len(x_test))*100))

# %69 -> SVR
# %67 -> çoklu lineer regresyon
# %67 -> KNN
# %76 YSA

# predict the test datas
df_test=pd.read_csv("test.csv")
    
df_test=df_test.drop(["PassengerId"],axis=1)
df_test=df_test.drop(["Name"],axis=1)
df_test=df_test.drop(["Ticket"],axis=1)
df_test=df_test.drop(["Fare"],axis=1)
df_test=df_test.drop(["Cabin"],axis=1)
df_test=df_test.drop(["Embarked"],axis=1)

df_test=pd.get_dummies(df_test,columns=["Sex"],prefix=["Sex"])
df_test=df_test.drop(["Sex_male"],axis=1)

df_test=df_test.dropna()
df_result=df_test
df_result["Survived"]=0
df_test=df_test.drop(["Survived"],axis=1)

df_test.index=np.arange(0,len(df_test),1)
for i in df_test.index:
    df_result["Survived"][i]=round(my_model.predict(pd.DataFrame(df_test.iloc[i]).T)[0])
    











































