import os
import glob
import shutil
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas
from xml.etree import ElementTree

# Funkcja dzielaca dane z folderu 'data' na zbior testowy oraz treningowy
def split_data(path):
    path_ann = os.path.join(path, 'annotations/*.xml')
    list_ann = glob.glob(path_ann)

    path_img = os.path.join(path, 'images/*.png')
    list_img = glob.glob(path_img)

    data = []
    data_speed = []
    data_nonespeed = []

    if(len(list_img) == len(list_ann)):
        for el in range(len(list_ann)):
            data.append({'annotation': list_ann[el], 'image': list_img[el], 'speedlimit': None})

    for el in data:
        parser = ElementTree.parse(el['annotation'])
        for object in parser.findall('object'):
            name = object.findall('name')
            for n in name:
                if(n.text == 'speedlimit'):
                    el['speedlimit'] = 1

    for el in data:
        if(el['speedlimit'] == 1):
            data_speed.append(el)
        else:
            data_nonespeed.append(el)

    path_train_ann = os.path.abspath('C:/Users/filip/Documents/GitHub/train/annotations')
    path_train_img = os.path.abspath('C:/Users/filip/Documents/GitHub/train/images')

    path_test_ann = os.path.abspath('C:/Users/filip/Documents/GitHub/test/annotations')
    path_test_img = os.path.abspath('C:/Users/filip/Documents/GitHub/test/images')

    # Podział danych na zbior testowy i treningowy
    # i = 1
    # for el in data_speed:
    #     if(i % 4 == 0):
    #         shutil.copy(el['annotation'], path_test_ann)
    #         shutil.copy(el['image'], path_test_img)
    #     else:
    #         shutil.copy(el['annotation'], path_train_ann)
    #         shutil.copy(el['image'], path_train_img)
    #     i += 1
    #
    # k = 1
    # for el in data_nonespeed:
    #     if (k % 4 == 0):
    #         shutil.copy(el['annotation'], path_test_ann)
    #         shutil.copy(el['image'], path_test_img)
    #     else:
    #         shutil.copy(el['annotation'], path_train_ann)
    #         shutil.copy(el['image'], path_train_img)
    #     k += 1
    return

# Funkcja wczytujace dane ze zbioru testowego i treningowego
def load_data(path):
    path_ann = os.path.abspath(os.path.join(path, 'annotations/*.xml'))
    path_img = os.path.abspath(os.path.join(path, 'images/*.png'))

    list_ann = glob.glob(path_ann)
    list_img = glob.glob(path_img)

    data = []
    if (len(list_img) == len(list_ann)):
        for el in range(len(list_ann)):
            data.append({'annotation': list_ann[el], 'image': cv2.imread(list_img[el])})

    return data

def main():
    # split_data('data')

    data_train = load_data('C:/Users/filip/Documents/GitHub/train')
    data_test = load_data('C:/Users/filip/Documents/GitHub/test')

    cv2.imshow('first train image', data_train[0]['image'])
    cv2.waitKey(0)
    return

if __name__ == '__main__':
    main()
