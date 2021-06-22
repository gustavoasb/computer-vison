import numpy as np
import cv2

def paintPixels(frame,mask,pixel):
    #Distancia euclediana
    mask = mask-pixel
    mask = np.square(mask)
    mask = np.sum(mask,axis=2)
    mask = np.sqrt(mask)
    mask = np.repeat(mask[...,None],3, axis=2)
    #Pintura distancia < 13
    frame = np.where(mask < 13, [0,0,255], frame)
    return frame

def mouseClick(event, x, y, flags, param):
    global original, pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        #Array rgb do pixel clicado
        pixel = np.array([int(original[y,x][0]),int(original[y,x][1]),int(original[y,x][2])])

def main():
    print('Press Q to stop the video')
    #Lendo webcam
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    global pixel, original
    pixel = [None, None, None]

    #Loop webcam
    while(cam.isOpened()):   
        ret, frame = cam.read()
        mask = frame.copy()
        original = frame.copy()
        cv2.setMouseCallback('webcam',mouseClick)
        if(pixel[0] != None):
            frame = paintPixels(frame,mask,pixel)
        cv2.imshow('webcam', frame.astype(np.uint8))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()