# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:36:19 2017

@author: YingJoy
"""

from sklearn import linear_model

reg = linear_model.Ridge(alpha = .5)
print reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print reg.coef_

print reg.intercept_