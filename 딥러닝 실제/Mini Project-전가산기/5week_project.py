# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 21:47:07 2021

@author: Jhung
"""

import numpy as np

def AND(x1, x2, print_flag = True):
    
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    
    if tmp <= 0:
        if print_flag == True: 
            print("[AND Gate] in :", x, ", out : ", 0)
        return 0
    else:
        if print_flag == True: 
            print("[AND Gate] in :", x, ", out : ", 1)
        return 1
    
def OR(x1, x2, print_flag = True):
    
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(w*x) + b
    
    if tmp <= 0:
        if print_flag == True: 
            print("[OR Gate] in :", x, ", out : ", 0)
        return 0
    else:
        if print_flag == True: 
            print("[OR Gate] in :", x, ", out : ", 1)
        return 1
    
def NAND(x1, x2, print_flag = True):
    
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    
    if tmp <= 0:
        if print_flag == True: 
            print("[NAND Gate] in :", x, ", out : ", 0)
        return 0
    else:
        if print_flag == True: 
            print("[NAND Gate] in :", x, ", out : ", 1)
        return 1
    
def NOR(x1, x2, print_flag = True):
    
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.1
    tmp = np.sum(w*x) + b
    
    if tmp <= 0:
        if print_flag == True: 
            print("[NOR Gate] in :", x, ", out : ", 0)
        return 0
    else:
        if print_flag == True: 
            print("[NOR Gate] in :", x, ", out : ", 1)
        return 1
    
def XOR(x1, x2, print_flag = True):
    s1 = NAND(x1, x2, False)
    s2 = OR(x1, x2, False)
    
    tmp = AND(s1, s2, False)
    if print_flag == True: 
        print("[XOR Gate] in : [", x1, x2, "], out : ", tmp)
    return tmp

def full_adder_function(x1, x2, x3, print_flag = True):
    
    s1 = XOR(x1, x2, False)
    s2 = AND(x1, x2, False)
    s3 = AND(s1, x3, False)
    
    Sum = XOR(s1, x3, False)
    Carry = OR(s3, s2, False)
    
    if print_flag == True:
        print("[FULL Addr Gate] in : [", x1, x2, x3, "], SUM : ", Sum, ", Carry : ", Carry)
        

full_adder_function(0, 0, 0)
full_adder_function(0, 0, 1)
full_adder_function(0, 1, 0)
full_adder_function(0, 1, 1)
full_adder_function(1, 0, 0)
full_adder_function(1, 0, 1)
full_adder_function(1, 1, 0)
full_adder_function(1, 1, 1)