# -*- coding: utf-8 -*-
"""
Created on Fri May 14 22:57:52 2021

@author: Jhung
"""

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np


# fetch_openml함수를 통해 mnist_784(손글씨) 데이터셋을 가져옴
mnist = fetch_openml('mnist_784')
# 가져온 mnist 데이터셋을 [0,1]로 정규화 진행 (원본 데이터는 0~255의 값을 갖음)
mnist.data = mnist.data/255.0
# 학습 데이터 6000개 저장 
X_train = mnist.data[:60000]
# 테스트 데이터 1000개 저장 (mnist_784의 총 데이터는 70000개)
X_test = mnist.data[60000:]
# 합습에 사용된 데이터의 결과를 정수형으로 저장
y_train = np.int16(mnist.target[:60000])
# 테스트에 사용된 데이터의 결과를 정수형으로 저장
y_test = np.int16(mnist.target[60000:])


# MLP 분류기 모델을 생성하는데 히든레이어의 사이즈를 100으로 하고 미니배치 크기를 512로 설정
# 경사하강법의 알고리즘 adam 적용, 에포크 최대 횟수 300, 학습률 초기값 0.001, 진행률 출력 Trun
mlp = MLPClassifier(hidden_layer_sizes=(50), learning_rate_init=0.001, batch_size=512, max_iter=300, solver='adam', verbose=True)
# 학습 데이터를 MLP 분류기 모델을 통해 학습 진행
mlp.fit(X_train, y_train)

# 테스트 데이터로 예측
res = mlp.predict(X_test)

# 10 x 10의 크기를 갖고 정수타입인  혼동 행렬 초기화
conf = np.zeros((10, 10), dtype=np.int16)
# 혼동 행렬 결과 생성 
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
# 혼동 행렬 결과 출력
print(conf)

# 정확률 계산
no_correct = 0
for i in range(10):
    no_correct += conf[i][i]
accuracy = no_correct / len(res)
print("테스트 집합에 대한 정확률은 : ", accuracy*100, "% 입니다.")

# 50 128 -> 3분54초
# 100 512 -> 3분 19초 