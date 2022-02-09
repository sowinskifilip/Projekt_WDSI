import os
import glob
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas
from xml.etree import ElementTree

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

    return


def main():
    split_data('data')
    return

if __name__ == '__main__':
    main()
