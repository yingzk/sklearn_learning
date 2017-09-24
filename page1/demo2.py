# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 22:34:38 2017

@author: YingJoy
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#导入糖尿病数据
diabetes = datasets.load_diabetes()

#只使用一个特征     np.newaxis相当于None
diabetes_X = diabetes.data[:, np.newaxis, 2]

#将数据分为训练和测试集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#将目标集分为训练和测试集
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

#创建线性回归对象
reg = linear_model.LinearRegression()

#用训练集训练模型
reg.fit(diabetes_X_train, diabetes_y_train)

#使用测试集做预测
diabetes_y_pred = reg.predict(diabetes_X_test)

#输出系数
print 'Coefficients: ', reg.coef_

#均方误差
print 'Mean squared error: %.2f' %mean_squared_error(diabetes_y_test, diabetes_y_pred)

#方差值，越接近1表面预测越好
print 'Variance score: %.2f' %r2_score(diabetes_y_test, diabetes_y_pred)

#画输出图
plt.scatter(diabetes_X_test, diabetes_y_test, c='k')
plt.plot(diabetes_X_test, diabetes_y_pred, c='b')

plt.xticks()
plt.yticks()

plt.show()