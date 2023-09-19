# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:13:18 2021

@author: admin
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
    
AND(0,0)
AND(1,0)
AND(0,1)
AND(1,1)
    
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
    
OR(0,0)
OR(1,0)
OR(0,1)
OR(1,1)
    
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
    
NAND(0,0)
NAND(1,0)
NAND(0,1)
NAND(1,1)
    
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
    
NOR(0,0)
NOR(1,0)
NOR(0,1)
NOR(1,1)

def XOR(x1, x2, print_flag = True):
    s1 = NAND(x1, x2, False)
    s2 = OR(x1, x2, False)
    
    tmp = AND(s1, s2, False)
    if print_flag == True: 
        print("[XOR Gate] in : [", x1, x2, "], out : ", tmp)
    return tmp

XOR(0,0)
XOR(1,0)
XOR(0,1)
XOR(1,1)