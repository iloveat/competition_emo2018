import os

root_dir = './data'


def extract_pure_video(org_video, new_video):
    command = 'ffmpeg -i %s -vcodec copy -an %s -y' % (org_video, new_video)
    os.system(command)


for dir_path, dir_names, file_names in os.walk(root_dir):
    for file_name in file_names:
        if len(file_name) < 5:
            continue
        if file_name[-4:] != '.avi':
            continue
        org_file_name = file_name
        org_file_path = os.path.join(dir_path, org_file_name)
        new_file_name = file_name.replace('.avi', '.mp4')
        new_file_path = os.path.join(dir_path, new_file_name)
        print org_file_path
        print new_file_path
        extract_pure_video(org_file_path, new_file_path)









