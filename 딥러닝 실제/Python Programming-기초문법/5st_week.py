# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:26:20 2021

@author: Jhung JoonYoung
"""
# =============================================================================
# 신경 회로망 
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    y = x > 0
    return y

x1 = np.arange(-3.0, 3.0, 0.1)
y1 = step_function(x1)

plt.plot(x1, y1)
plt.ylim(-0.1, 1.1)
plt.show()
    

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

x2 = np.arange(-5.0, 5.0, 0.1)
y2 = sigmoid_function(x2)

plt.plot(x2, y2)
plt.ylim(-0.1, 1.1)
plt.show()

def ReLU_function(x):
    return np.maximum(0, x)

x3 = np.arange(-5.0, 5.0, 0.1)
y3 = ReLU_function(x3)

plt.plot(x3, y3)
plt.ylim(-1, 5)
plt.show()

A = np.array([[1,2], [3,4], [5,6]])
print(A.shape)
B = np.array([7,8])
print(B.shape) 
print(np.dot(A, B))

