import pandas as pd
import os


def rename_dir(path: str = '../dataset/folder_num_class_map.txt'):
    classes_map = pd.read_table(path, header=None, sep=' ')
    classes_map.columns = ['folder', 'number', 'classes']
    class_dict ={}
    for i in range(len(classes_map)):
        class_dict[classes_map['folder'][i]] = f'{classes_map["number"][i]-1}-{classes_map["classes"][i]}'
    for dir, cls in class_dict.items():
        src = os.path.join('../dataset/', dir)
        dst = os.path.join('../dataset/', cls)
        try:
            os.rename(src, dst)
        except:
            pass