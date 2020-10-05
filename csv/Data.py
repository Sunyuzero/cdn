# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


##导入数据集
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , : -1 ].values
Y = dataset.iloc[ : , 3 ].values

##处理丢失数据
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

##数据分类
labelencoder_X = LabelEncoder()
labelencoder_Y = LabelEncoder()

X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

ct1 = ColumnTransformer([("country" , OneHotEncoder() , [1])], 'drop')
ct2 = ColumnTransformer([("country" , OneHotEncoder() , [1])], 'drop')
X = ct1.fit_transform(X).toarray()
Y = labelencoder_Y.fit_transform(Y)


##拆分数据集为训练集和测试集
X_train , X_test , Y_train , Y_test = train_test_split( X , Y , test_size = 0.2 , random_state = 0)
 

##特征量化
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_test)
