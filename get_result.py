import os
import numpy as np


root_dir = '/home/brycezou/DATA/emo_dataset/Test/result'


def path_to_label(array_path):
    if 'Angry' in array_path:
        label = 0
    elif 'Disgust' in array_path:
        label = 1
    elif 'Fear' in array_path:
        label = 2
    elif 'Happy' in array_path:
        label = 3
    elif 'Neutral' in array_path:
        label = 4
    elif 'Sad' in array_path:
        label = 5
    elif 'Surprise' in array_path:
        label = 6
    else:
        label = None
    return label


def idx_to_result(idx):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return emotions[idx]


for dir_path, dir_names, file_names in os.walk(root_dir):
    for file_name in file_names:
        if len(file_name) < 6:
            continue
        if file_name[-3:] != '1x7':
            continue
        file_path = os.path.join(dir_path, file_name)
        print file_path

        res_vec = np.fromfile(file_path, dtype=np.float32).reshape(1, 7)
        clz = np.argmax(res_vec)
        result = idx_to_result(clz)

        file_name = file_path.split('/result/')[1].split('mp4')[0]+'txt'
        print file_name, result

        with open(root_dir+'/'+file_name, 'w+') as f:
            f.write(result)




