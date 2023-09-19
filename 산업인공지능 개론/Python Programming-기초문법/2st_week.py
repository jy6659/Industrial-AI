# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("-----------------------------------")
for x in["철수", "영희", "아무게"]:
    print("이름 : ", x)

sum = 0

print("-----------------------------------")

for y in range(2, 10, 2):
    print(y, end = " ")
    sum = sum + y
    print(sum)
print(sum)

print("-----------------------------------")

sum = 0

limit = int(input("어디까지 계산할까요?:"))
print(limit, "까지 계산하겠습니다.")
for i in range(1, limit+1):
    print(i, end = " ")
    sum += i
    print(sum)
print("1부터 ", limit, "까지의 정수의 합 = ", sum)

print("-----------------------------------")

i = 0
while i<5 :
    print("환영합니다.")
    i = i + 1
print("반복문이 종료되었습니다.")

print("-----------------------------------")

i = 0
print("start")
while i<10 :
    print(i, end=" ")
    i += 1
print("end")

print("-----------------------------------")

i = 1
sum = 0

print("start")
while i<=10 :
    print(i, end=" ")
    sum = sum + i
    print("sum = ", sum)
    i += 1
    
print("end result : ", sum)

print("-----------------------------------")

i = 1
factorial = 1

while i <= 10 :
    factorial = factorial * i
    i += 1
print("10 fatorial is : %d" % factorial)

print("-----------------------------------")

i = 1
while i<10 :
    print("3 * %d = %d" % (i, 3*i))
    i += 1

print("-----------------------------------")

i = 1
sum = 0
while i <= 100 :
    if i % 3 == 0 : 
        sum = sum + i
    i += 1
print("3의 배수의 합은 : %d" % sum)

print("-----------------------------------")

def get_sum(start, end) :
    sum = 0
    for i in range(start, end+1) :
        sum = sum + i
    return sum

value = get_sum(1, 10)
print("get sum is : ", value)

print("-----------------------------------")

value = get_sum(21, 50)
print("get sum is : ", value)

print("-----------------------------------")

def FtoC(temp_f):
    temp_f = (5.0 * (temp_f - 32.0)) / 9.0
    return temp_f

temp_f = float(input("화씨 온도를 입력하시오 : "))
print(FtoC(temp_f))









    
