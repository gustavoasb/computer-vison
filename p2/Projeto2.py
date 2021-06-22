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
    

class Requisito1:
    def __init__(self):
        cv2.namedWindow("Video")
        cv2.setMouseCallback("Video",self.selecionaPixel)
        
        self.webcam = cv2.VideoCapture(0)

    def selecionaPixel(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if cliques.getClicks() == 0:
                pontoinicial.set(x, y)
                cliquestemp = cliques.getClicks() + 1
                cliques.set(cliquestemp)
                print("Coordenadas do primeiro Ponto = ({}, {})".format(x,y))

            elif cliques.getClicks() == 1:
                pontofinal.set(x, y)
                cliquestemp = cliques.getClicks() + 1
                cliques.set(cliquestemp)
                print("Coordenadas do segundo Ponto = ({}, {})".format(x,y))
                self.distanciaEuclidiana()

            else:
                print("Distancia Euclidiana ja calculada. Tecle 'q' para finalizar a execucao do programa")
                pass

    def distanciaEuclidiana(self):
        distancia = ((pontoinicial.getX()-pontofinal.getX())**2 + (pontoinicial.getY() - pontofinal.getY())**2)**(1/2.0)

        print("Distancia Euclidiana = {:.2f}".format(distancia))

    def run(self):
        while True:
            grab, imagem = self.webcam.read()
            if not grab:
                print("Erro")
                break

            if cliques.getClicks() > 1:
                    cv2.line(imagem,(pontoinicial.getX(),pontoinicial.getY()),(pontofinal.getX(),pontofinal.getY()), (0,165,255), 8)
                    ponto_texto = (int((pontoinicial.getX()+pontofinal.getX())/2), int((pontoinicial.getY()+pontofinal.getY())/2))
                    texto = "%.2f" % ((pontoinicial.getX()-pontofinal.getX())**2+(pontoinicial.getY()-pontofinal.getY())**2)**(1/2.)
                    cv2.putText(imagem, texto, ponto_texto, cv2.FONT_HERSHEY_DUPLEX, 0.8, (211,0,148), 2,cv2.LINE_AA)
                

            cv2.imshow("Video", imagem)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break

class Requisito3:
    def __init__(self):
        cv2.namedWindow("Video")
        self.webcam = cv2.VideoCapture(0)
        origin = [0,0,0]
        wcf = [2.8,2.8,0]

    def findPattern(webcam):
        while(1):


if __name__ == "__main__":
    pontoinicial = Point(0, 0)
    pontofinal = Point(0, 0)
    
    cliques = Clicks(0)
    
    Requisito1().run()