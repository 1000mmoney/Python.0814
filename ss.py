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
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


colums = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('./data/2.iris.csv', names=colums)

array = data.values
x = array[:, 0:4]
y = array[:, 4]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

# 모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, Y_train)
model.coef_
model.intercept_

# 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test)
MAE = mean_absolute_error(Y_test, y_pred)
print(MAE)


# 결과(모델 예측값 vs 실제값) 시각화
plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(X_test, Y_test, color='blue', label='Acual Values', marker='o')
plt.plot(X_test, y_pred, color='red', label='Predicted Values', marker='o')


# 그래프 레이블 및 타이틀 설정
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()