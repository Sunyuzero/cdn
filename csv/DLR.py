# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:04:35 2020

@author: zycjj
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

##导入数据集
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , : -1 ].values
Y = dataset.iloc[ : , 4 ].values


##数字化类别
labelencoder = LabelEncoder()
X[ : , 3 ] = labelencoder.fit_transform(X[ : , 3 ])
ct = ColumnTransformer([("state",OneHotEncoder(),[1])] , "drop" )
X = ct.fit_transform(X).toarray()


##躲避虚拟变量陷阱
X = X[ : , 1  : ]


##拆分数据集
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0 )


##训练多元线性回归模型
regressor = LinearRegression()
regressor.fit( X_train , Y_train )

Y_pred = regressor.predict(X_train)
print(Y_pred)


