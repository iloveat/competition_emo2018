import os
import numpy as np


root_dir = './data'


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


video_feature_map = dict()

for dir_path, dir_names, file_names in os.walk(root_dir):
    for file_name in file_names:
        if len(file_name) < 6:
            continue
        if file_name[-5:] != 'array':
            continue
        file_path = os.path.join(dir_path, file_name)
        # print file_path
        lbl = path_to_label(file_path)
        lbl_vec = np.zeros((7,), dtype=np.float32)
        lbl_vec[lbl] = 1
        print lbl, lbl_vec
        label_path = file_path[0:-5]+'label'
        lbl_vec.tofile(label_path)
        print label_path



