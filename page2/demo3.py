# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:50:44 2017

@author: YingJoy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#X是10x10的希尔伯特矩阵
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

#########################计算###########################
# 定义alpha的数量
n_alphas= 200
# 建立alphas向量    
# 对数等分 log以10为底  在-10 到-2之间分为n_alphas份   这里的alpha非常的小
alphas = np.logspace(-10, -2, n_alphas)

# 系数向量
coefs = []
# 循环alphas，一共n_alpha个岭回归模型，
# 分别以不同的alpha，训练, 最终保存所有模型计算出的系数到coefs向量中
for a in alphas:
    ridge = linear_model.Ridge(alpha = a, fit_intercept = False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
########################结束计算#########################

# 画图，显示结果   
# plt.gca()获取当前轴的对象ax，然后通过ax.plot()画图
ax = plt.gca()
ax.plot(alphas, coefs)
#将x轴取对数
ax.set_xscale('log')
#翻转x轴
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()