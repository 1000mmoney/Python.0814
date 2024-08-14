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

print(data.describe())

# array = data.values
# x = array[:, 0:4]
# y = array[:, 4]
#
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
#
# # 모델 선택 및 학습
# model = LinearRegression()
# model.fit(X_train, Y_train)
#
# # 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행
# y_pred = model.predict(X_test)
# print(confusion_matrix(y_pred, Y_test))
# print(classification_report(y_pred, Y_test))


