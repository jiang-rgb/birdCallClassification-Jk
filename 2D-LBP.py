import os
import numpy as np
import librosa
import soundfile as sound
from skimage.feature import hog
from skimage import data, color, exposure
import pandas as pd
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

radius = 5
n_points = 16


def LBP2(log_mel):

    lbp1 = local_binary_pattern(log_mel[0:12, :], n_points, radius, method='nri_uniform')
    lbp1 = lbp1.reshape(12*1379).astype(np.int64)
    lbp1 = np.bincount(lbp1, minlength=20)

    lbp2 = local_binary_pattern(log_mel[12:24, :], n_points, radius, method='nri_uniform')
    lbp2 = lbp2.reshape(12*1379).astype(np.int64)
    lbp2 = np.bincount(lbp2, minlength=20)

    lbp3 = local_binary_pattern(log_mel[24:36, :], n_points, radius, method='nri_uniform')
    lbp3 = lbp3.reshape(12*1379).astype(np.int64)
    lbp3 = np.bincount(lbp3, minlength=20)

    lbp4 = local_binary_pattern(log_mel[36:48, :], n_points, radius, method='nri_uniform')
    lbp4 = lbp4.reshape(12*1379).astype(np.int64)
    lbp4 = np.bincount(lbp4, minlength=20)

    lbp5 = local_binary_pattern(log_mel[48:60, :], n_points, radius, method='nri_uniform')
    lbp5 = lbp5.reshape(12*1379).astype(np.int64)
    lbp5 = np.bincount(lbp5, minlength=20)

    lbp6 = local_binary_pattern(log_mel[60:72, :], n_points, radius, method='nri_uniform')
    lbp6 = lbp6.reshape(12*1379).astype(np.int64)
    lbp6 = np.bincount(lbp6, minlength=20)

    lbp7 = local_binary_pattern(log_mel[72:84, :], n_points, radius, method='nri_uniform')
    lbp7 = lbp7.reshape(12*1379).astype(np.int64)
    lbp7 = np.bincount(lbp7, minlength=20)

    lbp8 = local_binary_pattern(log_mel[84:96, :], n_points, radius, method='nri_uniform')
    lbp8 = lbp8.reshape(12*1379).astype(np.int64)
    lbp8 = np.bincount(lbp8, minlength=20)

    lbp9 = local_binary_pattern(log_mel[96:108, :], n_points, radius, method='nri_uniform')
    lbp9 = lbp9.reshape(12*1379).astype(np.int64)
    lbp9 = np.bincount(lbp9, minlength=20)

    lbp10 = local_binary_pattern(log_mel[108:120, :], n_points, radius, method='nri_uniform')
    lbp10 = lbp10.reshape(12*1379).astype(np.int64)
    lbp10 = np.bincount(lbp10, minlength=20)

    aaa = np.hstack((lbp1, lbp2, lbp3, lbp4, lbp5, lbp6, lbp7, lbp8, lbp9, lbp10))

    return aaa


a = 243*10
n_mels = 120
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
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                               fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [0], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BAWW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [1], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BBWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [2], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BLBW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [3], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BLPW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [4], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BTBW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [5], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BTNW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [6], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BTYW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [7], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'BWWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [8], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CAWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [9], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CERW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [10], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CMWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [11], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'COLW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [12], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CONW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [13], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'COYE':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [14], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'CSWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [15], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'GCWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [16], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'GRWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [17], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'GWWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [18], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'HEWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [19], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'HOWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [20], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'KEWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [21], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'LOWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [22], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'LUWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [23], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'MAWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [24], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'NAWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [25], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'NOPA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [26], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'NOWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [27], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'OCWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [28], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'OVEN':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [29], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PAWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [30], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PIWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [31], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PRAW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [32], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'PROW':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [33], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'RFWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [34], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'TEWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [35], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'TOWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [36], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'VIWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [37], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'WEWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [38], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'WIWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [39], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'YEWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [40], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'YRWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [41], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)
        if file1 == 'YTWA':
            stereo, fs = sound.read(path)
            stereo = stereo - np.mean(stereo)
            stereo = stereo / np.max(np.abs(stereo))
            mel = librosa.feature.melspectrogram(stereo, sr=fs, n_fft=512, hop_length=32, n_mels=n_mels, fmin=0.0,
                                                 fmax=fs / 2, htk=True, norm=None)
            log_mel = np.log(mel + 1e-8)
            lbp1 = LBP2(log_mel)
            lbp1 = np.append(lbp1, [42], axis=0).reshape(1, a + 1)
            x = np.append(x, lbp1, axis=0)

x = pd.DataFrame(x)
x.to_csv("E:/csv3/2D-lbp/val1_b.csv", header=False, index=False)




