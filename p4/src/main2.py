import data
import features
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import VOCevalcls

limit = input("Insira a quantidade de imagens que você deseja utilizar: ")
limit = int(limit)

print("Lendo rotulos das imagens...")
a_train, a_test, a_val, train_ids, test_ids, val_ids, test_str = data.readAnnotations(limit)

print("Lendo imagens...")
images, n_images = data.readImages(limit)

print("Separando imagens de treinamento, teste e validação...")
im_train, im_test, im_val, n_train, n_test, n_val = data.separateImages(images,n_images,train_ids,test_ids,val_ids)

del images

print("Separando rótulos para cada tipo de imagem...")
labels_train = data.readLabels(a_train,n_train)
labels_teste = data.readLabels(a_test,n_test)
labels_val = data.readLabels(a_val,n_val)

print("Achando descritores das imagens de treinamento...")
sift = cv.xfeatures2d.SIFT_create()
descriptors_train_all = features.findDescriptors(im_train,n_train,sift)
descriptors_train = []
for i in range(n_train):
    des = descriptors_train_all[i]
    des = np.array(des)
    if len(des) > 400:
        des = des[0:400]
    descriptors_train.append(des)

print("Concatenando descritores...")
descriptors_train_complete = np.concatenate(descriptors_train,axis=0)
n_clusters = 150

print("Clusterizando...")
kmeans_obj = KMeans(n_clusters = n_clusters)
km = kmeans_obj.fit_predict(descriptors_train_complete)

print("Criando bag of visual words")
bag = []
count = 0
for i in range(n_train):
    a = km[count:count+len(descriptors_train[i])]
    hist, edges = np.histogram(a, bins = np.arange(n_clusters+1),density=True)
    bag.append(hist)
    count+=len(descriptors_train[i])

print("Treinando classificadores KNN")
classifier = []
for i in range(20):
    classifier_indiv = KNeighborsClassifier(n_neighbors=10)
    classifier_indiv.fit(bag,labels_train[i])
    classifier.append(classifier_indiv)

print("Achando descritores das imagens de teste")

sift2 = cv.xfeatures2d.SIFT_create()
descriptors_test = features.findDescriptors(im_test,n_test,sift2)
for i in range(n_test):
    des = descriptors_test[i]
    des = np.array(des)
    if len(des) > 400:
        des = des[0:400]
    descriptors_test[i] = des

print("Achando words das imagens de teste")
hist_test = []
results = []

for i in range(n_test):
    x = kmeans_obj.predict(descriptors_test[i])
    hist, edges = np.histogram(x, bins = np.arange(n_clusters+1),density=True)
    hist_test.append(hist)

print("Classificando images de teste")
results = []
for i in range(20):
    pred = classifier[i].predict_proba(hist_test)
    results.append(pred)

probs = np.zeros((n_test,20))
for i in range(n_test):
    for j in range(20):
        probs[i][j] = 1 - results[j][i][0] 

maximum = 0.0
for i in range(n_test):
    for j in range(20):
        if probs[i][j] > maximum:
            maximum = probs[i][j]
            img_class = j
    classe = data.classByIdx(img_class)       
    print("Imagem {}: Tem maior chance de conter um/uma {}\nProbabilidades: {}".format(test_ids[i],classe,probs[i])) 
    maximum = 0.0

print("1. Aeroplane, 2. Bicycle, 3.Bird, 4.Boat, 5.Bottle, 6.Bus, 7.Car, 8.Cat, 9.Chair, 10.Cow")
print("11.DinningTable, 12.Dog, 13.Horse, 14.Motorbike, 15.Person, 16.PottedPlant, 17.Sheep, 18.Sofa, 19.Train, 20.TVMonitor")
results_txt  = open("../VOCdevkit/results/VOC2007/Main/1_cls_person_car.txt", "w+") 
for i in range(n_test):
    results_txt.write(test_str[i])
    results_txt.write(" ")
    results_txt.write(str(probs[i][14]))
    if i != (n_test-1):
        results_txt.write("\n")

person_precision = VOCevalcls.get_AP("1","person","test",draw=True)
print(person_precision)