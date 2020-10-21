# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:51:14 2020

@author: zycjj
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

## 导入数据集
dataset =pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[ : , [2,3] ].values
Y = dataset.iloc[ : , 4 ].values

## 拆分数据集
X_train , X_test , Y_train , Y_test = train_test_split( X , Y , test_size = 0.2 , random_state = 0 )

## 特征缩放
# 归一化后加快了梯度下降求最优解的速度；
# 如果机器学习模型使用梯度下降法求最优解时，归一化往往非常有必要，否则很难收敛甚至不能收敛。
# 归一化有可能提高精度；
#一些分类器需要计算样本之间的距离（如欧氏距离），例如KNN。如果一个特征值域范围非常大，那么距离计算就主要取决于这个特征，从而与实际情况相悖（比如这时实际情况是值域范围小的特征更重要）。
sc = StandardScaler()
#fit方法是用于从一个训练集中学习模型参数，其中就包括了归一化时用到的均值，标准偏差。transform方法就是用于将模型用于位置数据，fit_transform就很高效的将模型训练和转化合并到一起，训练样本先做fit，得到mean，standard deviation，然后将这些参数用于transform（归一化训练数据），使得到的训练数据是归一化的，而测试数据只需要在原先得到的mean，std上来做归一化就行了，所以用transform就行了。
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## 逻辑回归应用于训练集
classifier = LogisticRegression()
classifier.fit( X_train , Y_train ) 

## 预测结果
Y_pred = classifier.predict( X_test )

## 生成混淆矩阵
# 混淆矩阵（confusion matrix）衡量的是一个分类器分类的准确程度
cm = confusion_matrix( Y_test , Y_pred )

## 可视化
X_set , Y_set = X_train , Y_train
X1 , X2 = meshgrid( np.arrange( start = X_set[ : , 0 ].min() - 1 , stop = X_set[ : , 0 ].max() + 1 ,step = 0.01 , 
                    np.arrange( start = X_set[ : , 1 ].min() - 1 , stop = X_set[ : , 1 ].max() + 1 ,step = 0.01 )
plt.contourf(X1 , X2 , classfier.predict( np.array([X1.ravel()])))
                   