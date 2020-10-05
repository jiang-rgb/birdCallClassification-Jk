import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import soundfile as sound
import pandas as pd
from skimage.feature import local_binary_pattern
import pywt
from scipy.fftpack import fft, dct

D = 'sym4'


def lbp(stereo):
    length = len(stereo)
    fea = np.empty(shape=[1], dtype=int)
    for k in range(0, length - 9 + 1):
        aaa = 0
        group = stereo[k: k + 9]
        center = group[4]
        for j in range(9):
            if j != 4:
                if group[j] >= center:
                    if j == 0:
                        aaa += 128
                    if j == 1:
                        aaa += 64
                    if j == 2:
                        aaa += 32
                    if j == 3:
                        aaa += 16
                    if j == 5:
                        aaa += 8
                    if j == 6:
                        aaa += 4
                    if j == 7:
                        aaa += 2
                    if j == 8:
                        aaa += 1
        fea = np.append(fea, aaa)
    fea = np.delete(fea, [0])
    hist = np.bincount(fea, minlength=256)
    # print(hist)

    std = np.std(stereo)
    thr = std * 0.5
    LBP_LT = np.empty(shape=[1], dtype=int)
    for k in range(0, length - 9 + 1):
        aaa = 0
        group = stereo[k: k + 9]
        center = group[4]
        for j in range(9):
            if j != 4:
                if (group[j] - center) < -thr:
                    if j == 0:
                        aaa += 128
                    if j == 1:
                        aaa += 64
                    if j == 2:
                        aaa += 32
                    if j == 3:
                        aaa += 16
                    if j == 5:
                        aaa += 8
                    if j == 6:
                        aaa += 4
                    if j == 7:
                        aaa += 2
                    if j == 8:
                        aaa += 1
        LBP_LT = np.append(LBP_LT, aaa)
    LBP_LT = np.delete(LBP_LT, [0])
    hist_LT = np.bincount(LBP_LT, minlength=256)
    # print(hist_LT)

    LBP_UT = np.empty(shape=[1], dtype=int)
    for k in range(0, length - 9 + 1):
        aaa = 0
        group = stereo[k: k + 9]
        center = group[4]
        for j in range(9):
            if j != 4:
                if (group[j] - center) > thr:
                    if j == 0:
                        aaa += 128
                    if j == 1:
                        aaa += 64
                    if j == 2:
                        aaa += 32
                    if j == 3:
                        aaa += 16
                    if j == 5:
                        aaa += 8
                    if j == 6:
                        aaa += 4
                    if j == 7:
                        aaa += 2
                    if j == 8:
                        aaa += 1
        LBP_UT = np.append(LBP_UT, aaa)
    LBP_UT = np.delete(LBP_UT, [0])
    hist_UT = np.bincount(LBP_UT, minlength=256)
    # print(hist_UT)

    aaa = np.hstack((hist, hist_LT, hist_UT))

    return aaa


def dwt(stereo):
    aaa = lbp(stereo)
    # 第一级
    ca, cd = pywt.dwt(stereo, D)
    aaa1 = lbp(ca)
    # 第二级
    ca, cd = pywt.dwt(ca, D)
    aaa2 = lbp(ca)
    # 第一级
    ca, cd = pywt.dwt(ca, D)
    aaa3 = lbp(ca)
    # 第一级
    ca, cd = pywt.dwt(ca, D)
    aaa4 = lbp(ca)
    # 第一级
    ca, cd = pywt.dwt(ca, D)
    aaa5 = lbp(ca)
    # 第一级
    ca, cd = pywt.dwt(ca, D)
    aaa6 = lbp(ca)
    # 第一级
    ca, cd = pywt.dwt(ca, D)
    aaa7 = lbp(ca)
    # 第一级
    ca, cd = pywt.dwt(ca, D)
    aaa8 = lbp(ca)
    # 第一级
    ca, cd = pywt.dwt(ca, D)
    aaa9 = lbp(ca)
    wav = np.hstack((aaa, aaa1, aaa2, aaa3, aaa4, aaa5, aaa6, aaa7, aaa8, aaa9))

    return wav


def all(stereo):
    a1 = dwt(stereo)
    a2 = dwt(abs(fft(dct(stereo))))
    wav = np.hstack((a1, a2))
    return wav


# a: 特征数量
a = 7680*2
# 文件夹名
filepath1 = "E:/CLO43/2s/validation_data/fold_1/"  # 添加路径
filename1 = os.listdir(filepath1)  # 得到文件夹下的所有文件名称
x = np.empty(shape=[0, a + 1], dtype=float)
for file1 in filename1:
    filepath = filepath1 + file1 + "/"  # 类别路径
    filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
    for file in filename:
        path = filepath + file  # 类别下的文件
        stereo, fs = sound.read(path)
        if file1 == 'AMRE':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [0], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BAWW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [1], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BBWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [2], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BLBW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [3], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BLPW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [4], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BTBW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [5], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BTNW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [6], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BTYW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [7], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BWWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [8], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CAWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [9], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CERW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [10], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CMWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [11], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'COLW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [12], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CONW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [13], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'COYE':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [14], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CSWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [15], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'GCWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [16], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'GRWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [17], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'GWWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [18], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'HEWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [19], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'HOWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [20], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'KEWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [21], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'LOWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [22], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'LUWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [23], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'MAWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [24], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'NAWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [25], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'NOPA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [26], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'NOWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [27], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'OCWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [28], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'OVEN':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [29], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PAWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [30], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PIWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [31], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PRAW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [32], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PROW':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [33], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'RFWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [34], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'TEWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [35], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'TOWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [36], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'VIWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [37], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'WEWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [38], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'WIWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [39], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'YEWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [40], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'YRWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [41], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'YTWA':
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            lbp1 = all(stereo)
            lbp1 = lbp1.reshape(a)
            lbp1 = np.append(lbp1, [42], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)

x = pd.DataFrame(x)
x = x.fillna(0)
x.to_csv("E:/csv2/lbp11/val1_a6.csv", header=False, index=False)






