# -*- coding:utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread


class Profile:
    def __init__(self):
        self.name = "Hong Gil Dong"
        self.number = 2021000000
        
    def setName(self, myName):
        self.name = myName
        
    def setNumber(self, myNumber):
        self.number = myNumber
        
    def setImage(self, imagePath):
        self.image = imread(imagePath)
        
    def printProfile(self):
        title = "Number : " + self.number + ", Name : " + self.name 
        plt.title(title)
        plt.imshow(self.image)
        plt.show()

filePath = 'web.jpg'
    
myProfile = Profile()
myProfile.setName("Jhung Joon Young")
myProfile.setNumber("2021254002")
myProfile.setImage(filePath)
myProfile.printProfile()
