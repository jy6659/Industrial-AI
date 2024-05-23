# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print("-----------------------------")

print("Hello World!!!")

print("-----------------------------")

print("type : ", type(10.0))
print("type : ", type(2.55))
print("type : ", type("10"))

print("-----------------------------")

x = 10
print(x)
x = 100
print(x)
y = 3.14
print(y)

print("-----------------------------")

a = [1, 2, 3, 4, 5]
print(a)
print("length : ", len(a))
print("list 1 : ", a[0])
print("slicing : ", a[0:2])
print("slicing : ", a[:3])
print("slicing : ", a[3:])
print("slicing : ", a[2:-2])

print("-----------------------------")

me = {'height':180}
print("my height : ", me['height'])
me['weist'] = 70
print("my weist : ", me['weist'])
print(me)

print("-----------------------------")

hungry = True
sleepy = False

print("hungry type : ", type(hungry), " Value : ", hungry)
print("sleepy tpye : ", type(sleepy), " Value : ", sleepy)

print("hungry type : ", type(not hungry), " Value : ", not hungry)
print("hungry or sleepy: ", hungry or sleepy)

print("-----------------------------")

hungry = True
if hungry:
    print("I'm hungry")
    
hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")

print("-----------------------------")

for i in [1, 2, 3]:
    print("loop : ", i)

print("-----------------------------")

def hello():
    print("hello world!!!")

hello()
    
def hello(object):
    print("Hello " + object + "!")
    
hello("Jhung")

print("-----------------------------")

class Man:
    def __init__(self, name):
        self.name = name
        print("Initialzed!")        
    def Hollo(self):
        print("Hello ", self.name + "!")
    def Goodby(self):
        print("goodby " + self.name + "!")


m1 = Man("David")
m1.Hollo()
m1.Goodby()

m2 = Man("Judy")
m2.Hollo()
m2.Goodby()

print("-----------------------------")

import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

print(x-y)
print(x+y)
print(x*y)
print(x/y)
print(x/2)

print("-----------------------------")

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[3.0, 0.0], [0.0, 6.0]])

print(A)
print("data type : ", A.dtype)
print("Shape A : ", A.shape)

print ("A + B = ", A+B)
print("A*B = ", A*B)
print("A*10 = ", A*10)

print("-----------------------------")

C = np.array([[1.0, 2.0], [3.0, 4.0]])
D = np.array([10.0, 20.0])
print("C*D = ", C*D)

print("-----------------------------")

print(A[0])
print(A[1,1])

