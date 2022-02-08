import os
import glob
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas
from xml.etree import ElementTree

def load_data(path):
    # path_ann = os.path.join(path, 'annotations')
    # list_ann = os.listdir(path_ann)

    path_img = os.path.join(path, 'images')

    path_ann = os.path.join(path, 'annotations/*.xml')
    list_ann = glob.glob(path_ann)


    names = []

    for el in list_ann:
        dom = ElementTree.parse(el)
        names.append(dom.findall('object/name'))

    print(names)
    return


def main():
    load_data('data')
    return

if __name__ == '__main__':
    main()
