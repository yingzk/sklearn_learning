# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:38:38 2017

@author: yzk13
"""

from sklearn import linear_model

clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])

print clf.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print clf.alpha_