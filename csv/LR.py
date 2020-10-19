# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:45:50 2020

@author: zycjj
"""
import pandas as pd
import numpy as ny
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 导入数据集
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : , : 1 ].values
Y = dataset.iloc[ : , 1 ].values

## 拆分数据集
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split( X , Y , test_size = 1/4 , random_state = 0 )

## 使用简单线性回归模型训练数据集
regressor = LinearRegression()
regressor = regressor.fit( X_train , Y_train )

## 预测结果
Y_pred = regressor.predict( X_test )
print(Y_pred)

## 结果可视化
plt.scatter ( X_train , Y_train , color = 'red' )
plt.plot ( X_train , regressor.predict(X_train) , color = 'blue' )
plt.show() 

plt.scatter( X_test , Y_test ,color = 'red' )
plt.plot( X_test , Y_test , color = 'blue' )
plt.show()