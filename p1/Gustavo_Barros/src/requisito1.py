import numpy as np
import cv2

def mouseClick(event, x, y, flags, param):
    #Checa se a ação realizada é um click com o botao esquerdo do mouse
    if event == cv2.EVENT_LBUTTONDOWN: 
        pixelInfo(y,x,img)

#Imprime informaçoes do pixel que entrou como parametro na função
def pixelInfo(x,y,img):
    #É necessario checar se a imagem está em tons de cinza, pois isso influencia no
    #que precisa ser mostrado.
    print('Row:{0} Column:{1}'.format(x,y))
    if grayscale == False:
        print('R:{0} G:{1} B:{2}'.format(img[x,y][2], img[x,y][1], img[x,y][0]))
    else:
        print('Intesity:{0}'.format(img[x,y]))

def main():
    print('Press any key to close the image')
    global img, grayscale
    
    img = cv2.imread('..//data//kiminonawa.jpg',-1)
    #Testa se a imagem está em tons de cinza ou nao e armazena na variavel Grayscale.
    #Imagens com cor tem mais de 2 canais
    if len(img.shape) > 2:
        grayscale = False
    else:
        grayscale = True
        
    cv2.imshow('image',img)
    
    #Função que espera ações do mouse e chama a função de click
    cv2.setMouseCallback("image", mouseClick)
    cv2.waitKey(0)
    cv2.destroyAllWindows()     