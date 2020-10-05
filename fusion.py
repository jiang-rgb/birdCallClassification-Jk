import numpy as np
from sklearn.metrics import balanced_accuracy_score

# 1.读取数据集
path = 'C:/Users/16697/Desktop/final fusion1/1D-lbp5.csv'
lbp1D = np.loadtxt(path, dtype=float, delimiter=',')
print(lbp1D.shape)

path = 'C:/Users/16697/Desktop/final fusion1/2D-lbp5.csv'
lbp2D = np.loadtxt(path, dtype=float, delimiter=',')
print(lbp2D.shape)

path = 'E:/csv3/2D-lbp/test5_LBP2D.csv'
test1 = np.loadtxt(path, dtype=float, delimiter=',')
print(test1.shape)


# 2.划分数据与标签
x_test, y_test = np.split(test1, indices_or_sections=(2430,), axis=1)  # x为数据，y为标签

y_sum = lbp1D + lbp2D
y_predict = np.argmax(y_sum, axis=1)
balanced_accuracy = balanced_accuracy_score(y_test, y_predict)
print(balanced_accuracy)

