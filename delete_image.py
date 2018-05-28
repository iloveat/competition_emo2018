import os

root_dir = './emo_dataset'


def delete_image(dir_name):
    command = 'rm %s/*.jpg' % dir_name
    os.system(command)


for dir_path, dir_names, file_names in os.walk(root_dir):
    delete_image(dir_path)





