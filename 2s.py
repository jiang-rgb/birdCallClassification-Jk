import os
from pydub import AudioSegment
# 文件夹名
filepath1 = "E:/CLO43/split_data/CLO43_preprocess_data/test_data/fold_2/"  # 添加路径
filename1 = os.listdir(filepath1)  # 得到文件夹下的所有文件名称
for file1 in filename1:
    filepath = filepath1 + file1 + "/"  # 类别路径
    filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
    for file in filename:
        path = filepath + file  # 类别下的文件
        audio = AudioSegment.from_file(path, "wav")  # 读取文件
        audio1 = audio
        for i in range(200):
            audio += audio1
        audio2 = audio[:2000]
        if audio2.frame_count() != 44100:
            print(audio2.frame_count())
        audio2.export("E:/CLO43/2.1s/test_data/fold_2/" + file1 + "/" + file, format="wav")






