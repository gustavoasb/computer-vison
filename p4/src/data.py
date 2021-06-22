import cv2 as cv
import numpy as np
import glob

def readAnnotations(limit):
    filenames_annotations_train = [annot_train for annot_train in glob.glob("../VOCdevkit/VOC2007/ImageSets/Main/*_train.txt")]
    filenames_annotations_test = [annot_test for annot_test in glob.glob("../VOCdevkit/VOC2007/ImageSets/Main/*_test.txt")]
    filenames_annotations_val = [annot_val for annot_val in glob.glob("../VOCdevkit/VOC2007/ImageSets/Main/*_val.txt")]
    annotations_test = []
    annotations_train = []
    annotations_val = []
    train_ids = []
    val_ids = []
    test_ids = []
    test_ids_str = []

    with open("../VOCdevkit/VOC2007/ImageSets/Main/train.txt") as file:
        train = file.read().splitlines()
    for i in range(len(train)):
        x = int(train[i])
        if(x>limit):
            break
        train_ids.append(x)

    
    with open("../VOCdevkit/VOC2007/ImageSets/Main/test.txt") as file:
        test = file.read().splitlines()
    for i in range(len(test)):
        x = int(test[i])
        if(x>limit):
            break
        test_ids.append(x)
        test_ids_str.append(test[i])


    with open("../VOCdevkit/VOC2007/ImageSets/Main/val.txt") as file:
        val = file.read().splitlines()
    for i in range(len(val)):
        x = int(val[i])
        if(x>limit):
            break
        val_ids.append(x)

    for annot_train in filenames_annotations_train:
        annot_r_tr = open(annot_train,"r")
        annotations_lines = annot_r_tr.read().splitlines()
        annotations_train.append(annotations_lines)
    
    for annot_test in filenames_annotations_test:
        annot_r_tr = open(annot_test,"r")
        annotations_lines = annot_r_tr.read().splitlines()
        annotations_test.append(annotations_lines)

    for annot_val in filenames_annotations_val:
        annot_r_tr = open(annot_val,"r")
        annotations_lines = annot_r_tr.read().splitlines()
        annotations_val.append(annotations_lines)

    return annotations_train, annotations_test, annotations_val, train_ids, test_ids, val_ids, test_ids_str

def readImages(limit):
    filenames = [img for img in glob.glob("../VOCdevkit/VOC2007/JPEGImages/*.jpg")]
    images = []
    n_images = 0

    for img in filenames:
        n = cv.imread(img, 0)
        images.append(n)
        n_images+=1
        if n_images == limit:
            break

    return images, n_images

def separateImages(images, n_images, train, test, val):
    images_train = []
    images_test = []
    images_val = []
    n_train = len(train)
    n_test = len(test)
    n_val = len(val)

    for i in range(n_train):
        index = train[i]
        images_train.append(images[index-1])

    for i in range(n_test):
        index = test[i]
        images_test.append(images[index-1])

    for i in range(n_val):
        index = val[i]
        images_val.append(images[index-1])

    return images_train, images_test, images_val, n_train, n_test, n_val

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

def classByIdx(index):
    if index == 0:
        classe = "Avião"
    elif index == 1:
        classe = "Bicicleta"
    elif index == 2:
        classe = "Pássaro"
    elif index == 3:
        classe = "Barco"
    elif index == 4:
        classe = "Garrafa"
    elif index == 5:
        classe = "Ônibus"
    elif index == 6:
        classe = "Carro"
    elif index == 7:
        classe = "Gato"
    elif index == 8:
        classe = "Cadeira"
    elif index == 9:
        classe = "Vaca"
    elif index == 10:
        classe = "Mesa de Jantar"
    elif index == 11:
        classe = "Cachorro"
    elif index == 12:
        classe = "Cavalo"
    elif index == 13:
        classe = "Moto"
    elif index == 14:
        classe = "Pessoa"
    elif index == 15:
        classe = "Vaso de Planta"
    elif index == 16:
        classe = "Ovelha"
    elif index == 17:
        classe = "Sofá"
    elif index == 18:
        classe = "Trem"
    else:
        classe = "Monitor de TV"
    return classe
    
