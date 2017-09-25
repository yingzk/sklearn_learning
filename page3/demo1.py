# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:31:15 2017

@author: YingJoy
"""

from sklearn import linear_model

reg = linear_model.Lasso(alpha = 0.1)
print reg.fit([[0, 0], [1, 1]], [0, 1])

print reg.predict([[1, 1]])