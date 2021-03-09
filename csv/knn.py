# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:47:01 2020

@author: zycjj
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

## 导入数据集并拆分为测试训练集
dataset = pd.read_csv('Social_Network_Ads.csv') 
X = dataset.iloc[ : , [2 , 3]].values
Y = dataset.iloc[ : , 4 ].values
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.1 , random_state = 0)

## 特征缩放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## k-NN训练
classfier = KNeighborsClassifier(n_neighbors = 10 , metric = "minkowski" , p = 2)
classfier.fit(X_train , Y_train)

## 预测
Y_pred = classfier.predict(X_test)
print(Y_pred)
print("=========================")
print(Y_test)
