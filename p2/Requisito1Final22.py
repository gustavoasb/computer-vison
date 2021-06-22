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
n_linhas = 0

print ('\033[1m'+'\033[37m'+'Teclque "q" a qualquer momento para finalizar a execucao\n'+'\033[0;0m')


def Requisito1(event, x, y, flags, param):
    global pontoinicial
    global pontofinal
    global cliques
    global frame
    global n_linhas
    global linhas

    if event == cv2.EVENT_LBUTTONDOWN:
        if cliques.getClicks() == 0:
            print ('\033[1m'+'\033[35m'+'+--------------------------------------------------+'+'\033[0;0m')
            print ('\033[1m'+'\033[37m'+'|    Reta {}                   X        Y'.format(n_linhas + 1)+'\033[0;0m')
            print ('\033[1m'+'\033[37m'+'+--------------------------------------------------+'+'\033[0;0m')
            pontoinicial.set(x, y)
            cliques.set(1)
            print ('\033[1m'+'\033[37m'+'|    Ponto inicial            {}      {}'.format(x, y)+'\033[0;0m')

        elif cliques.getClicks() == 1:
            pontofinal.set(x, y)
            print ('\033[1m'+'\033[37m'+'|    Ponto final              {}      {}'.format(x, y)+'\033[0;0m')
            print ('\033[1m'+'\033[37m'+'+--------------------------------------------------+'+'\033[0;0m')
            distancia = ((pontoinicial.getX() - pontofinal.getX())**2 +
                         (pontoinicial.getY() - pontofinal.getY())**2)**(1/2.0)
            print ('\033[1m'+'\033[37m'+'|    Distancia Euclidiana          {:.2f}'.format(distancia)+'\033[0;0m')
            print ('\033[1m'+'\033[35m'+'+--------------------------------------------------+\n\n'+'\033[0;0m')
            cliques.set(0)
            linhas.append(n_linhas)
            linhas[n_linhas] = (pontoinicial.getX(),pontoinicial.getY(),pontofinal.getX(),pontofinal.getY())
            n_linhas = n_linhas+1
            #print(linhas)
            #print(n_linhas)

def drawLines(frame,linhas):
    global n_linhas
    #if n_linhas > (-1):
    for x in range(0,n_linhas):
        cv2.line(frame,(linhas[x][0],linhas[x][1]),(linhas[x][2],linhas[x][3]),(0,165,255),8)
        ponto_texto = (int((linhas[x][0] + linhas[x][2])/2), int((linhas[x][1] + linhas[x][3])/2))
        texto = "%.2f" % ((linhas[x][0] - linhas[x][2])** 2+(linhas[x][1] - linhas[x][3])**2)**(1/2.0)
        cv2.putText(frame, texto, ponto_texto, cv2.FONT_HERSHEY_DUPLEX, 0.8, (211, 0, 148), 2, cv2.LINE_AA)

linhas = []                        
cam = cv2.VideoCapture(0)
cv2.namedWindow('webcam')
while(cam.isOpened()):   
    ret, frame = cam.read()
    cv2.setMouseCallback('webcam',Requisito1)
    drawLines(frame,linhas)
    cv2.imshow('webcam', frame.astype(np.uint8))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows




