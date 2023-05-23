from glob import glob
import os
import shutil
import pickle

from PIL import Image
import pandas as pd
from tqdm import tqdm


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def save_data(paths, save_dir, mode):
    os.makedirs(os.path.join(save_dir, f'{mode}/images'), exist_ok=True)

    image_id = 0
    labels = []
    for data_path in paths:
        batch = unpickle(data_path)
        labels.extend(batch[b'labels'])
        for data in tqdm(batch[b'data']):
            Image.fromarray(data.reshape(32, 32, 3)).save(os.path.join(save_dir, f'{mode}/images', f'{image_id}.jpg'))
            image_id += 1

    label_df = pd.DataFrame(labels, columns=['label'])
    label_df.to_csv(os.path.join(save_dir, f'{mode}_labels.csv'))


def save_all_data(data_dir, save_dir):
    train_paths = glob(os.path.join(data_dir, 'data_batch_*'))
    test_paths = [os.path.join(data_dir, 'test_batch')]
    
    save_data(train_paths, save_dir, 'train')
    save_data(test_paths, save_dir, 'test')        


# if __name__ == '__main__':
#     save_all_data('pytorch-course/classification/cifar10/data/cifar-10-batches-py', 'pytorch-course/classification/cifar10/image_data')