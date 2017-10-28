#Script for parsing dataset : https://raw.githubusercontent.com/mttk/STL10/master/stl10_input.py

from __future__ import print_function

import sys
import os, sys, tarfile
import numpy as np
import matplotlib.pyplot as plt

import keras
    
if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib # ugly but works
else:
    import urllib

print(sys.version_info) 

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'


def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    image = np.reshape(image, (3, 96, 96))
    image = np.transpose(image, (2, 1, 0))
    return image


def download_and_extract():
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

        
def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return keras.utils.to_categorical(labels)[:,1:]

    
def load():
    files = os.path.join(DATA_DIR,'stl10_binary')
    cross_val_indices = []
    with open(os.path.join(files, "fold_indices.txt"), "r") as ins:
        for line in ins:
            cross_val_indices.append([int(x) for x in line.rstrip().split(' ')])
    X_train = read_all_images(os.path.join(files,"train_X.bin")).astype('float32')
    Y_train = read_labels(os.path.join(files,"train_y.bin"))
    X_test = read_all_images(os.path.join(files,"test_X.bin")).astype('float32')
    Y_test = read_labels(os.path.join(files,"test_y.bin"))
    X_train /= 255.
    X_test /= 255.
    return (X_train, Y_train), (X_test, Y_test), cross_val_indices

def unlabelled_data():
    files = os.path.join(DATA_DIR,'stl10_binary')
    unlabelled_x = read_all_images(os.path.join(files,"unlabeled_X.bin")).astype('float32')
    unlabelled_x /= 255.
    return unlabelled_x

if __name__ == "__main__":
    download_and_extract()
