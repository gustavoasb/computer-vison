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

    if event == cv2.EVENT_LBUTTONDOWN:
        if cliques.getClicks() == 0:
            pontoinicial.set(x, y)
            cliques.set(1)
            print("Coordenadas do primeiro Ponto = ({}, {})".format(x, y))

        elif cliques.getClicks() == 1:
            pontofinal.set(x, y)
            print("Coordenadas do segundo Ponto = ({}, {})".format(x, y))

            cv2.line(imagem, (pontoinicial.getX(), pontoinicial.getY()),
                     (pontofinal.getX(), pontofinal.getY()), (0, 165, 255), 8)
            ponto_texto = (int((pontoinicial.getX() + pontofinal.getX())/2),
                           int((pontoinicial.getY() + pontofinal.getY())/2))
            texto = "%.2f" % ((pontoinicial.getX() - pontofinal.getX())
                              ** 2+(pontoinicial.getY() - pontofinal.getY())**2)**(1/2.0)
            cv2.putText(imagem, texto, ponto_texto,
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (211, 0, 148), 2, cv2.LINE_AA)
            distancia = ((pontoinicial.getX() - pontofinal.getX())**2 +
                         (pontoinicial.getY() - pontofinal.getY())**2)**(1/2.0)
                         
            print("Distancia Euclidiana = {:.2f}".format(distancia))
            cliques.set(0)


arquivo = input("Digite o nome do arquivo: ")
imagem = cv2.imread(arquivo)
cv2.namedWindow("Trabalho")
cv2.setMouseCallback("Trabalho", Requisito1)


while True:
    cv2.imshow("Trabalho", imagem)
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break
