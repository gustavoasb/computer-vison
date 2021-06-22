import numpy as np
import cv2 

#Função que pinta os pixels e faz a distancia euclediana ou absoluta
def paintPixels(img,mask,pixel):
    global grayscale
    if(grayscale == False):
        mask = mask-pixel
        mask = np.square(mask)       #Diferença euclediana
        mask = np.sum(mask,axis=2)
        mask = np.sqrt(mask)
        #As alteracoes deixam a mask com apenas 1 canal de cor, entao precisamos
        #deixar com 3 para podermos utiliza-lo
        mask = np.repeat(mask[...,None],3, axis=2)
        #Pintando de vermelho lugares com DE menor que 13
        img = np.where(mask < 13, [0,0,255], img)
        return img
    else:
        mask = mask - pixel #Diferença absoluta entre matrizes
        mask = np.abs(mask)
        #Pintando de vermelho lugares com DA menor que 13
        img = np.where(mask < 13, [0,0,255], img)
        return img

#Função de ação do mouse
def mouseClick(event, x, y, flags, param):
    global img, mask, original, pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        #Formando array RGB do pixel clicado
        pixel = np.array([int(original[y,x][0]),int(original[y,x][1]),int(original[y,x][2])])
        img = paintPixels(img,mask,pixel)
        cv2.imshow('image', img.astype(np.uint8))
        #Temos que voltar a imagem ao normal depois de fazermos as alteraçoes
        img = original.copy()

def main():
    print('Press any key to close the image')
    global original, img, mask, grayscale
    img = cv2.imread('..//data//kiminonawa.jpg',-1)

    #Testa se a imagem está em tons de cinza ou não
    if len(img.shape) > 2:
        #Caso a imagem tenha ALPHA temos que eliminar esse canal e manter só RGB
        if img.shape[2] == 4:
            img = img[:,:,:3] #Descarta alpha
            grayscale = False
        else:
            grayscale = False
    else:
        #Caso esteja em tons de cinza precisamos que esteja em RGB para pintar
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        grayscale = True

    #Imagem para fazer as operacoes com matrix
    mask = img.copy()
    #Imagem para podermos voltar á imagem original
    original = img.copy()
    cv2.imshow('image', img.astype(np.uint8))
    cv2.setMouseCallback('image',mouseClick)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
