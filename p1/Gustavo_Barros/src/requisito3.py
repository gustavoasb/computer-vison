import numpy as np
import cv2

#Função que pinta os pixels e faz a distancia euclediana
def paintPixels(frame,mask,pixel):
    mask = mask-pixel
    mask = np.square(mask)
    mask = np.sum(mask,axis=2)
    mask = np.sqrt(mask)
    mask = np.repeat(mask[...,None],3, axis=2)
    frame = np.where(mask < 13, [0,0,255], frame)
    return frame

#Função de ação do mouse
def mouseClick(event, x, y, flags, param):
    global pixel, original
    if event == cv2.EVENT_LBUTTONDOWN:
        #Formando array RGB do pixel clicado
        pixel = np.array([int(original[y,x][0]),int(original[y,x][1]),int(original[y,x][2])])

def main():
    print('Press Q to stop the video')

    video = cv2.VideoCapture('..//data//neverland.avi') #Lendo vídeo de entrada

    global pixel
    pixel = [None, None, None]
    global original
    #Loop do vídeo
    while(video.isOpened()):   
        ret, frame = video.read() #Lendo frames do vídeo
        mask = frame.copy() #Mascara pra calculos
        original = frame.copy() #Cópia do frame original
        cv2.setMouseCallback('frame',mouseClick) #Função do mouse
        if(pixel[0] != None):
            frame = paintPixels(frame,mask,pixel) #Função de pintura
        cv2.imshow('frame', frame.astype(np.uint8)) #Mostra frame atualizado
        if cv2.waitKey(25) & 0xFF == ord('q'): #Fecha imagem a cada 25 ms
            break                              #Q para parar
    video.release()
    cv2.destroyAllWindows()