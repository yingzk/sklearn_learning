# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 17:34:39 2017

@author: YingJoy
"""

#导入线性模型
from sklearn import linear_model

#reg为线性回归
reg = linear_model.LinearRegression()
print '````````````````````````````````````````'

#对输入输出进行拟合
print reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2]) 
print '````````````````````````````````````````'

#系数矩阵(模型的权重)
print reg.coef_
print '````````````````````````````````````````'

#训练后模型截距
print reg.intercept_
print '````````````````````````````````````````'

#训练后模型预测
print reg.predict([[2, 5]])
print '````````````````````````````````````````'

#训练是否标准化
print reg.normalize
print '````````````````````````````````````````'

#获取模型训练前设置的参数
print reg.get_params
print '````````````````````````````````````````'
