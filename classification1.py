import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn import tree
import time


# 1.读取数据集
path = 'E:/csv2/lbp11/train3_a6.csv'
train1 = np.loadtxt(path, dtype=float, delimiter=',')
print(train1.shape)

path = 'E:/csv2/lbp11/val3_a6.csv'
val1 = np.loadtxt(path, dtype=float, delimiter=',')
print(val1.shape)

path = 'E:/csv3/2D-lbp/test3_LBP1D.csv'
test1 = np.loadtxt(path, dtype=float, delimiter=',')
print(test1.shape)

data_train = np.vstack((train1, val1))
print(data_train.shape)

# 2.划分数据与标签
x_train, y_train = np.split(data_train, indices_or_sections=(7680*2,), axis=1)  # x为数据，y为标签
x_test, y_test = np.split(test1, indices_or_sections=(7680*2,), axis=1)  # x为数据，y为标签


time_start = time.time()  # 开始计时
x_train = x_train[:, ::2]
x_test = x_test[:, ::2]

nca = NeighborhoodComponentsAnalysis(random_state=42, n_components=100, init='pca')


std = MinMaxScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)


nca.fit(x_train, y_train)
x_train1 = nca.transform(x_train)
x_test1 = nca.transform(x_test)


time_end = time.time()    # 结束计时
time_c = time_end - time_start   # 运行所花时间
print('time cost', time_c, 's')

# 3.训练svm分类器
C_list = [40]
kernel_list = ['rbf']
x = [['kernel', 'c', 'gamma', 'acc']]
for kernel_index in range(len(kernel_list)):
    for C_index in range(len(C_list)):
        classifier = svm.SVC(C=C_list[C_index], kernel=kernel_list[kernel_index], probability=True)
        classifier.fit(x_train1, y_train.ravel())
        score = balanced_accuracy_score(y_test, classifier.predict(x_test1))
        aaa = [kernel_list[kernel_index], C_list[C_index], score]
        print(aaa)
        a = classifier.predict_proba(x_test1)
        a = pd.DataFrame(a)
        a.to_csv("C:/Users/16697/Desktop/final fusion1/1D-lbp5.csv", header=False, index=False)


