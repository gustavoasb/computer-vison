import cv2
import numpy as np

fs = cv2.FileStorage("output0.xml", cv2.FILE_STORAGE_READ)
fread = fs.getNode("intrinsics")
data = fread.mat()
origin = [0,0,0]
datainv = np.linalg.inv(data)
opt = datainv * [1,1,1] 
print(opt)