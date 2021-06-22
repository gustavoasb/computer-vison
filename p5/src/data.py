import cv2 as cv
import numpy as np
import glob

def readAnnotations(limit):
    filenames_annotations_train = [annot_train for annot_train in glob.glob("../VOCdevkit/VOC2007/ImageSets/Main/*_trainval.txt")]
    filenames_annotations_test = [annot_test for annot_test in glob.glob("../VOCdevkit/VOC2007/ImageSets/Main/*_test.txt")]
    annotations_test = []
    annotations_train = []
    train_ids = []
    train_s_ids = []
    test_ids = []
    test_s_ids = []
    test_ids_str = []

    with open("../VOCdevkit/VOC2007/ImageSets/Main/trainval.txt") as file:
        train = file.read().splitlines()
    for i in range(len(train)):
        x = int(train[i])
        if(x>limit):
            break
        train_ids.append(x)

    with open("../VOCdevkit/VOC2007/ImageSets/Segmentation/trainval.txt") as file:
        train = file.read().splitlines()
    for i in range(len(train)):
        x = int(train[i])
        if(i>limit):
            break
        train_s_ids.append(x)
    
    with open("../VOCdevkit/VOC2007/ImageSets/Main/test.txt") as file:
        test = file.read().splitlines()
    for i in range(len(test)):
        x = int(test[i])
        if(x>limit):
            break
        test_ids.append(x)
        test_ids_str.append(test[i])

    with open("../VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt") as file:
        test = file.read().splitlines()
    for i in range(len(test)):
        x = int(test[i])
        if(i>limit):
            break
        test_s_ids.append(x)

    seg_ids = train_s_ids+test_s_ids
    seg_ids.sort()
    print(seg_ids)

    for annot_train in filenames_annotations_train:
        annot_r_tr = open(annot_train,"r")
        annotations_lines = annot_r_tr.read().splitlines()
        annotations_train.append(annotations_lines)
    
    for annot_test in filenames_annotations_test:
        annot_r_tr = open(annot_test,"r")
        annotations_lines = annot_r_tr.read().splitlines()
        annotations_test.append(annotations_lines)

    return annotations_train, annotations_test, train_ids, train_s_ids, test_ids, test_s_ids, test_ids_str

def readImages(limit):
    filenames = [img for img in glob.glob("../VOCdevkit/VOC2007/JPEGImages/*.jpg")]
    filenames_segmented = [img2 for img2 in glob.glob("../VOCdevkit/VOC2007/SegmentationClass/*.png")]
    images = []
    images_segmented = []
    n_images = 0
    n_images_segmented = 0

    for img in filenames:
        n = cv.imread(img, 0)
        images.append(n)
        n_images+=1
        if n_images == limit:
            break

    for img2 in filenames_segmented:
        n = cv.imread(img)
        images_segmented.append(n)
        n_images_segmented+=1
        if n_images_segmented == limit:
            break

    return images, n_images, images_segmented, n_images_segmented

def separateImages(images, images_segmented, n_images, train, train_s, test, test_s):
    images_train = []
    images_test = []
    images_val = []
    images_train_segmented = []
    n_train = len(train)
    n_test = len(test)

    for i in range(n_train):
        index = train[i]
        images_train.append(images[index-1])

    for i in range(n_test):
        index = test[i]
        images_test.append(images[index-1])

    return images_train, images_test, images_val, images_train_segmented, n_train, n_test

def readLabels(annotations,n_images):
    n_labels = 20
    labels = np.zeros((20,n_images))
    for i in range(n_labels):
        for j in range(n_images):
            x = annotations[i][j][7:9]
            if x == ' 1':
                x = 1
            elif x == ' 0':
                x = 0
            else:
                x = -1
            labels[i][j] = x
    return labels