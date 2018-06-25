import os
import numpy as np


root_dir = './data'


video_feature_map = dict()

for dir_path, dir_names, file_names in os.walk(root_dir):
    for file_name in file_names:
        if len(file_name) < 5:
            continue
        if file_name[-4:] != '1024':
            continue
        key = file_name.split('.mp4')[0]+'.mp4'
        key = os.path.join(dir_path, key)
        # print key
        value = file_name.split('.mp4')[1]
        # print value
        if key in video_feature_map:
            video_feature_map[key].append(value)
        else:
            video_feature_map[key] = [value]

for k, v in video_feature_map.items():
    v.sort()

# print video_feature_map


video_data_map = dict()

for k, v in video_feature_map.items():
    video_data_map[k] = []
    for file_name in v:
        file_path = k + file_name
        # print file_path
        data = np.fromfile(file_path, dtype=np.float32)
        # print k, data.shape, data[0]
        video_data_map[k].append(data)

for k, v in video_data_map.items():
    # print k
    # for data in v:
    #     print data[0]
    video_data_map[k] = np.array(v)


for k, v in video_data_map.items():
    print k, v.shape
    file_path = k+'_array'
    v.tofile(file_path)





















