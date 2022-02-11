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
            data.append({'annotation': list_ann[el], 'image': cv2.imread(list_img[el]), 'speedlimit': None})
    else:
        print('Wrong data set!')

    return data

# def generate_random_frame(image_height, image_width):
#     y_min = random.randint(0, image_height)
#     x_min = random.randint(0, image_width)
#     y_max = random.randint(y_min, image_height)
#     x_max = random.randint(x_min, image_width)
#
#     height = y_max - y_min
#     width = x_max - x_min
#
#     dimensions = [height, width]
#
#     return x_min, y_min, x_max, y_max, dimensions
#
# def isRectangleOverlap(R1, R2):
#     if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
#         return False
#     else:
#         return True

def crop_images(data):
    cropped_data = []
    # i = 1
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

            # xmin_rand, ymin_rand, xmax_rand, ymax_rand, [height_rand, width_rand] = generate_random_frame(height, width)
            #
            # frame_A = [xmin, ymin, xmax, ymax]
            # frame_B = [xmin_rand, ymin_rand, xmax_rand, ymax_rand]
            # intersection = isRectangleOverlap(frame_A, frame_B)
            #
            # while((width_rand < 0.1 * width or height_rand < 0.1 * height) and intersection == True):
            #     xmin_rand, ymin_rand, xmax_rand, ymax_rand, [height_rand, width_rand] = generate_random_frame(height, width)
            #     frame_B = [xmin_rand, ymin_rand, xmax_rand, ymax_rand]
            #     intersection = isRectangleOverlap(frame_A, frame_B)


            # rand_img = el['image'][ymin_rand:ymax_rand, xmin_rand:xmax_rand]
            # cv2.imshow('rand', rand_img)
            # cv2.waitKey(0)

            # if(i % 50 == 0):
                # k += 1
                # cropped_data.append({'image': rand_img, 'label': 0})

            # i += 1

            # if(i == 565):
            #     rand_rect = cv2.rectangle(el['image'], (xmin_rand, ymin_rand), (xmax_rand, ymax_rand), (255, 0, 0), -1)
            #     rect = cv2.rectangle(el['image'], (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)
            #     cv2.imshow(str(intersection), rect)
            #     cv2.waitKey(0)

    return cropped_data

def learn_bovw(data):
    dict_size = 128
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

def getInput():
    input_data = []
    classify = input("Type 'classify' to start testing:")
    while(classify != 'classify'):
        classify = input('Try again:')

    n_files = input('Number of files to test:')
    for i in range(0, int(n_files)):
        file = input('Name of the file:')
        n_images = input('Number of images to test:')
        for k in range(0, int(n_images)):
            cordinates = input('Cordinates (xmin xmax ymin ymax):').split()
            for i in range(0, len(cordinates)):
                cordinates[i] = int(cordinates[i])
            input_data.append({'filename': file, 'cordinates': cordinates})

    return input_data

def prepare_test_data(path, data):
    data_prepared = []

    path_img = os.path.join(path, 'images')
    for el in data:
        path_img = os.path.abspath(os.path.join(path_img, el['filename']))
        img = cv2.imread(path_img)
        xmin, xmax, ymin, ymax = el['cordinates']
        img = img[ymin:ymax, xmin:xmax]
        data_prepared.append({'image': img})
        path_img = path_img = os.path.join(path, 'images')

    return data_prepared

def predict(rf, data):
    for sample in data:
        if sample['desc'] is not None:
            predict = rf.predict(sample['desc'])
            if(int(predict) == 1):
                print('speedlimit')
            else:
                print('other')
    return

def main():
    # split_data('data')

    path_train = 'C:/Users/filip/Documents/GitHub/train'
    path_test = 'C:/Users/filip/Documents/GitHub/test'

    data_train = load_data(path_train)
    # data_test = load_data(path_test)

    data_train = crop_images(data_train)

    print('learning BoVW')
    learn_bovw(data_train)

    print('extracting train features')
    data_train = extract_features(data_train)

    print('training')
    rf = train(data_train)

    input_data = getInput()
    data_test = prepare_test_data(path_test, input_data)

    print('extracting test features')
    data_test = extract_features(data_test)

    predict(rf, data_test)



    return

if __name__ == '__main__':
    main()
