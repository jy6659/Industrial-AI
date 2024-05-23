# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:07:54 2021

@author: Jhung
"""

print("-----------------------------")

#scores = []
#for i in range(10):
#    scores.append(int(input("성적을 입력하시오 : ")))
#    print(len(scores))
#print(scores)

print("-----------------------------")

scores3 = [32,56,64,72,12,37,98,77,59,69]

for element in scores3:
    print(element)

print("-----------------------------")

STUDENTS = 5
scores4 = []
scoreSum = 0
for i in range(STUDENTS):
    value = int(input("성적을 입력하시오 : "))
    scores4.append(value)
    scoreSum += value
    print(len(scores4), "번째 성적 입력 값 : ", value)
    
heighValue = 0;
for i in range(STUDENTS):
    if 80 <= scores4[i] :
        heighValue += 1

    
print("성적 평균값은 : ", scoreSum / STUDENTS)
print("80점 이상의 학생은 : ", heighValue, " 입니다.")  

print("-----------------------------")  