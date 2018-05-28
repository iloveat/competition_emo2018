import os

root_dir = './data'


def detect_face_in_video(video_path, face_size=224):
    command = './face_demo %s %d 0' % (video_path, face_size)
    os.system(command)


for dir_path, dir_names, file_names in os.walk(root_dir):
    for file_name in file_names:
        if len(file_name) < 5:
            continue
        if file_name[-4:] != '.mp4':
            continue
        org_file_name = file_name
        org_file_path = os.path.join(dir_path, org_file_name)
        detect_face_in_video(org_file_path)











