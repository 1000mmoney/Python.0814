# matplotlib 설치
import matplotlib.pyplot as plt

#pandas 설치
import pandas as pd

# numpy 설치
import numpy as np

# scikit-learn 설치
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 각 항목 이름 붙이기

data = pd.read_csv('./data/1.salary.csv')

# 바꾸기 할 항목 범위 정하기
array = data.values
X = array[:, 0:2]

Y = array[:, -1]

# 소수점 아래로 바꾸기 1, 데이터 전처리 : Min-Max 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.3)

# 모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, Y_train)

# 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test)
print(y_pred)

# 예측값을 0 또는 1로 변환 (임계값 설정 필요)
y_pred_binary = (y_pred > 0.5).astype(int)
print(y_pred_binary)

# 예측 정확도 계산
accuracy = accuracy_score(Y_test, y_pred_binary)
print(accuracy)

# 결과(모델 예측값 vs 실제값) 시각화
plt.figure(figsize=(10, 6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Salary Values', marker='o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='Experience Values', marker='x')

# 그래프 레이블 및 타이틀 설정
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
