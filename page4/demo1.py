# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:54:26 2017

@author: YingJoy
"""

from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

#预测分类
print (clf.predict([[2, 2]]))

# 预测被为某类的概率
print (clf.predict_proba([[2., 2.]]))