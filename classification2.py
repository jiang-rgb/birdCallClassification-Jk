import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import time


# 1.读取数据集
path = 'E:/csv3/2D-lbp/train1_b.csv'
train1 = np.loadtxt(path, dtype=float, delimiter=',')
print(train1.shape)

path = 'E:/csv3/2D-lbp/val1_b.csv'
val1 = np.loadtxt(path, dtype=float, delimiter=',')
print(val1.shape)

path = 'E:/csv3/2D-lbp/test1_LBP2D.csv'
test1 = np.loadtxt(path, dtype=float, delimiter=',')
print(test1.shape)

data_train = np.vstack((train1, val1))
print(data_train.shape)

# 2.划分数据与标签
x_train, y_train = np.split(data_train, indices_or_sections=(2430,), axis=1)  # x为数据，y为标签
x_test, y_test = np.split(test1, indices_or_sections=(2430,), axis=1)  # x为数据，y为标签

std = MinMaxScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

nca = NeighborhoodComponentsAnalysis(random_state=42, n_components=200, init='pca')
nca.fit(x_train, y_train)
x_train1 = nca.transform(x_train)
x_test1 = nca.transform(x_test)


classifier = svm.SVC(C=5, kernel='rbf', probability=True)
classifier.fit(x_train1, y_train.ravel())
score = balanced_accuracy_score(y_test, classifier.predict(x_test1))
print(score)
a = classifier.predict_proba(x_test1)
a = pd.DataFrame(a)
a.to_csv("C:/Users/16697/Desktop/final fusion1/2D-lbp1.csv", header=False, index=False)

