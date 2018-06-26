import os


root_dir = './data'


os.system('rm -f train_list.txt')
os.system('rm -f valid_list.txt')


for dir_path, dir_names, file_names in os.walk(root_dir):
    for file_name in file_names:
        if len(file_name) < 6:
            continue
        if file_name[-5:] != 'label':
            continue
        file_path = os.path.join(dir_path, file_name)
        print file_path

        if 'validation' in file_path:
            with open('valid_list.txt', 'a+') as fv:
                fv.write('%s\n' % file_path[0:-6])
        else:
            with open('train_list.txt', 'a+') as ft:
                ft.write('%s\n' % file_path[0:-6])














