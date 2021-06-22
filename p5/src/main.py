import data
import numpy as np
import cv2 as cv
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

limit = input("Insira a quantidade de imagens que você deseja utilizar: ")
limit = int(limit)

print("Lendo rotulos das imagens...")
a_train, a_test, train_ids, train_s_ids, test_ids, test_s_ids, test_str = data.readAnnotations(limit)

print("Lendo imagens...")
images, n_images, images_segmented, n_segmented = data.readImages(limit)

print("Separando imagens de treinamento, teste e validação...")
im_train, im_test, im_val, im_s_train, n_train, n_test = data.separateImages(images,images_segmented,n_images,train_ids,train_s_ids,test_ids,test_s_ids)

