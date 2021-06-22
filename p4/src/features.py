def findDescriptors(images,n_images,sift):
    descriptors = []
    for i in range(n_images):
        kp, des = sift.detectAndCompute(images[i], None)
        descriptors.append(des)
    return descriptors
