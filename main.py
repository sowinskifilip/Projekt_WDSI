import os
import glob
import shutil
import random
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas
from xml.etree import ElementTree

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.855 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0


# TODO Skuteczność detekcji (0/2)

# TODO Poprawki po terminie. (-1)

# Split dataset to training and test datasets
# def split_data(path):
#     path_ann = os.path.join(path, 'annotations/*.xml')
#     list_ann = glob.glob(path_ann)
#
#     path_img = os.path.join(path, 'images/*.png')
#     list_img = glob.glob(path_img)
#
#     data = []
#     data_speed = []
#     data_nonespeed = []
#
#     if(len(list_img) == len(list_ann)):
#         for el in range(len(list_ann)):
#             data.append({'annotation': list_ann[el], 'image': list_img[el], 'speedlimit': None})
#
#     for el in data:
#         parser = ElementTree.parse(el['annotation'])
#         for object in parser.findall('object'):
#             name = object.findall('name')
#             for n in name:
#                 if(n.text == 'speedlimit'):
#                     el['speedlimit'] = 1
#
#     for el in data:
#         if(el['speedlimit'] == 1):
#             data_speed.append(el)
#         else:
#             data_nonespeed.append(el)
#
#     path_train_ann = os.path.abspath('C:/Users/filip/Documents/GitHub/train/annotations')
#     path_train_img = os.path.abspath('C:/Users/filip/Documents/GitHub/train/images')
#
#     path_test_ann = os.path.abspath('C:/Users/filip/Documents/GitHub/test/annotations')
#     path_test_img = os.path.abspath('C:/Users/filip/Documents/GitHub/test/images')
#
#     Export splited data
#     i = 1
#     for el in data_speed:
#         if(i % 4 == 0):
#             shutil.copy(el['annotation'], path_test_ann)
#             shutil.copy(el['image'], path_test_img)
#         else:
#             shutil.copy(el['annotation'], path_train_ann)
#             shutil.copy(el['image'], path_train_img)
#         i += 1
#
#     k = 1
#     for el in data_nonespeed:
#         if (k % 4 == 0):
#             shutil.copy(el['annotation'], path_test_ann)
#             shutil.copy(el['image'], path_test_img)
#         else:
#             shutil.copy(el['annotation'], path_train_ann)
#             shutil.copy(el['image'], path_train_img)
#         k += 1
#     return

# Load data from train dataset
def load_data(path):
    path_ann = os.path.abspath(os.path.join(path, 'annotations/*.xml'))
    path_img = os.path.abspath(os.path.join(path, 'images/*.png'))

    list_ann = glob.glob(path_ann)
    list_img = glob.glob(path_img)

    list_ann.sort()
    list_img.sort()

    data = []
    if (len(list_img) == len(list_ann)):
        for el in range(len(list_ann)):
            data.append({'annotation': list_ann[el], 'image': cv2.imread(list_img[el]), 'speedlimit': None})
    else:
        print('Wrong data set!')

    return data

# Generate random rectangle // not working with BoVW
# def generate_random_frame(image_height, image_width):
#     max_size = 100
#     min_size = 40
#
#     y_min = random.randint(0, image_height)
#     x_min = random.randint(0, image_width)
#     y_max = random.randint(y_min, y_min + max_size)
#     x_max = random.randint(x_min, x_min + max_size)
#
#     height = y_max - y_min
#     width = x_max - x_min
#
#     while(height < min_size or height > max_size or y_max > image_height):
#         y_max = random.randint(y_min, y_min + max_size)
#         height = y_max - y_min
#
#     while (width < min_size or width > max_size or x_max > image_width):
#         x_max = random.randint(x_min, x_min + max_size)
#         width = x_max - x_min
#
#     return x_min, y_min, x_max, y_max

# Check intersection between 2 rectangles
# def isRectangleOverlap(R1, R2):
#     if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
#         return False
#     else:
#         return True

# Crop images from train dataset
def crop_images(data):
    cropped_data = []
    # i = 0
    # k = 0
    for el in data:
        height, width, _ = el['image'].shape

        parser = ElementTree.parse(el['annotation'])
        for object in parser.findall('.//object'):

            xmin = int(object.find('bndbox/xmin').text)
            ymin = int(object.find('bndbox/ymin').text)
            xmax = int(object.find('bndbox/xmax').text)
            ymax = int(object.find('bndbox/ymax').text)

            cropped_img = el['image'][ymin:ymax, xmin:xmax]
            height_cut, width_cut, _ = cropped_img.shape

            name = object.find('name').text
            if (name == 'speedlimit'):
                el['speedlimit'] = 1
            else:
                el['speedlimit'] = 0

            if(width_cut > 0.1 * width and height_cut > 0.1 * height):
                cropped_data.append({'image': cropped_img, 'label': el['speedlimit']})

            # Generate random image // not working
            # if(i % 50 == 0):
            #     xmin_rand, ymin_rand, xmax_rand, ymax_rand = generate_random_frame(height, width)
            #     frame_A = [xmin, ymin, xmax, ymax]
            #     frame_B = [xmin_rand, ymin_rand, xmax_rand, ymax_rand]
            #     intersection = isRectangleOverlap(frame_A, frame_B)
            #     while(intersection == True):
            #         xmin_rand, ymin_rand, xmax_rand, ymax_rand = generate_random_frame(height, width)
            #         frame_B = [xmin_rand, ymin_rand, xmax_rand, ymax_rand]
            #         intersection = isRectangleOverlap(frame_A, frame_B)
            #
            #     rand_img = el['image'][ymin_rand:ymax_rand, xmin_rand:xmax_rand]
            #     # cv2.imshow('rand', rand_img)
            #     # cv2.waitKey(0)
            #     cropped_data.append({'image': rand_img, 'label': 0})

    return cropped_data

# Clusterization - Bag of Visual Words
def learn_bovw(data):
    dict_size = 256
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)
    return

# Extracting features from data - matching descriptors
def extract_features(data):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        sample['desc'] = desc
    return data

# Train model - RandomForestClassifier
def train(data):
    descs = []
    labels = []
    for sample in data:
        if sample['desc'] is not None:
            descs.append(sample['desc'].squeeze(0))
            labels.append(sample['label'])
    rf = RandomForestClassifier()
    rf.fit(descs, labels)

    return rf

# Read input data from console
def getInput():
    input_data = []
    classify = input()
    while(classify != 'classify'):
        classify = input()

    n_files = input()
    for i in range(0, int(n_files)):
        file = input()
        n_images = input()
        for k in range(0, int(n_images)):
            cordinates = input().split()
            for i in range(0, len(cordinates)):
                cordinates[i] = int(cordinates[i])
            input_data.append({'filename': file, 'cordinates': cordinates})

    return input_data

# Prepare test data for prediction
def prepare_test_data(path, data):
    data_prepared = []

    path_img = os.path.join(path, 'images')
    for el in data:
        path_img = os.path.abspath(os.path.join(path_img, el['filename']))
        img = cv2.imread(path_img)
        xmin, xmax, ymin, ymax = el['cordinates']
        img = img[ymin:ymax, xmin:xmax]
        data_prepared.append({'image': img})
        # cv2.imshow('Input frame', img)
        # cv2.waitKey(0)
        path_img = path_img = os.path.join(path, 'images')

    return data_prepared

# Return model prediction in console
def predict(rf, data):
    for sample in data:
        if sample['desc'] is not None:
            predict = rf.predict(sample['desc'])
            if(int(predict) == 1):
                print('speedlimit')
            else:
                print('other')
        else:
            print('other')
    return

def main():
    # split_data('data')

    # Load path from dataset
    path_train = "../train"
    path_test = "../test"

    data_train = load_data(path_train)

    data_train = crop_images(data_train)

    # print('learning BoVW')
    learn_bovw(data_train)

    # print('extracting train features')
    data_train = extract_features(data_train)

    # print('training')
    rf = train(data_train)

    input_data = getInput()
    data_test = prepare_test_data(path_test, input_data)

    data_test = extract_features(data_test)

    predict(rf, data_test)

    return

if __name__ == '__main__':
    main()
