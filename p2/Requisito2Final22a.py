import cv2
import numpy as np


class Point:
    def __init__(self, x, y):
        self.set(x, y)

    def set(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


class Clicks:
    def __init__(self, x):
        self.set(x)

    def set(self, x):
        self.x = x

    def getClicks(self):
        return self.x


pontoinicial = Point(0, 0)
pontofinal = Point(0, 0)
cliques = Clicks(0)


def Requisito1(event, x, y, flags, param):
    global pontoinicial
    global pontofinal
    global cliques
    global frame
    global n_linhas
    global linhas

    if event == cv2.EVENT_LBUTTONDOWN:
        if cliques.getClicks() == 0:
            pontoinicial.set(x, y)
            cliques.set(1)
            print("Coordenadas do primeiro Ponto = ({}, {})".format(x, y))

        elif cliques.getClicks() == 1:
            pontofinal.set(x, y)
            print("Coordenadas do segundo Ponto = ({}, {})".format(x, y))
            distancia = ((pontoinicial.getX() - pontofinal.getX())**2 +
                         (pontoinicial.getY() - pontofinal.getY())**2)**(1/2.0)
            print("Distancia Euclidiana = {:.2f}".format(distancia))
            cliques.set(0)
            linhas.append(n_linhas)
            linhas[n_linhas] = (pontoinicial.getX(),pontoinicial.getY(),pontofinal.getX(),pontofinal.getY())
            n_linhas = n_linhas+1

def drawLines(frame,linhas):
    global n_linhas
    #if n_linhas > (-1):
    for x in range(0,n_linhas):
        cv2.line(frame,(linhas[x][0],linhas[x][1]),(linhas[x][2],linhas[x][3]),(0,165,255),8)
        ponto_texto = (int((linhas[x][0] + linhas[x][2])/2), int((linhas[x][1] + linhas[x][3])/2))
        texto = "%.2f" % ((linhas[x][0] - linhas[x][2])** 2+(linhas[x][1] - linhas[x][3])**2)**(1/2.0)
        cv2.putText(frame, texto, ponto_texto, cv2.FONT_HERSHEY_DUPLEX, 0.8, (211, 0, 148), 2, cv2.LINE_AA)

n_linhas = 0
linhas = []                        
cam = cv2.VideoCapture(0)
cv2.namedWindow('raw')
cv2.namedWindow('distorted')
while(cam.isOpened()):   
    ret, frame = cam.read()
    cv2.setMouseCallback('raw',Requisito1)
    cv2.setMouseCallback('distorted',Requisito1)
    drawLines(frame,linhas)
    cv2.imshow('raw', frame.astype(np.uint8))
    cv2.imshow('distorted', frame.astype(np.uint8))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows




