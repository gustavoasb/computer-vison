import numpy as np
import cv2
import time
import xml.etree.cElementTree as ET



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

def calculaNormaTvecs(tvecs):
    distance = np.zeros(5)
    for x in range(0,5):
        distance[x] = ((tvecs[x][0]*tvecs[x][0])+(tvecs[x][1]*tvecs[x][1])+(tvecs[x][2]*tvecs[x][2]))**(1/2)
        print("Norma {}: {:.2f}cm".format(x,distance[x]/10))
    print("Norma media = {:.2f}cm".format(np.mean(distance/10)))
    print("Desvio padrao = {:.2f}cm".format(np.std(distance/10)))
    return distance

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
            linhas[n_linhas] = (pontoinicial.getX(),pontoinicial.getY(),pontofinal.getX(),pontofinal.getY(), n_linhas + 1)
            n_linhas = n_linhas+1

def drawLines(frame,linhas):
    global n_linhas
    #if n_linhas > (-1):
    for x in range(0,n_linhas):
        cv2.line(frame,(linhas[x][0],linhas[x][1]),(linhas[x][2],linhas[x][3]),(0,165,255),8)
        ponto_texto = (int((linhas[x][0] + linhas[x][2])/2), int((linhas[x][1] + linhas[x][3])/2))
        texto = str(linhas[x][4])
        cv2.putText(frame, texto, ponto_texto, cv2.FONT_HERSHEY_DUPLEX, 0.8, (211, 0, 148), 2, cv2.LINE_AA)


n_linhas = 0 #Armazena o numero de linhas criadas
linhas = [] #Array de linhas

# Linhas e colunas do Tabuleiro de Xadrez do Pattern
board_w = 8
board_h = 6

# Linhas e colunas do Tabuleiro de Xadrez do Pattern (tamanho real) em CM
board_rw = 0.0
board_rh = 0.0
board_rw = input("Insira o tamanho da largura de um retangulo no chessboard (em MM). Recomendado: 28\n")
board_rh = input("Insira o tamanho da altura de um retangulo no chessboard(em MM). Recomendado: 28\n")
if((board_rw.isnumeric() == False) or (board_rh.isnumeric() == False)):
    print("Tamanho nao definido, encerrando programa")
    exit()
board_rw = float(board_rw)
board_rh = float(board_rh)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_h*board_w, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
realobjp = np.zeros((board_h*board_w, 3), np.float32)
realobjp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# prepare object points using MM given
for x in range(0,48):
    realobjp[x][0] = realobjp[x][0]*board_rw
    realobjp[x][1] = realobjp[x][1]*board_rh

# Arrays to store object points and image points from all the images.
realobjpoints = []  # 3d point in real world space
objpoints = []  # 3d point in unit world space
imgpoints = []  # 2d points in image plane.

captura = cv2.VideoCapture(0)
print ('\033[1m'+'\033[37m'+'Teclque "q" a qualquer momento para finalizar a execucao\n'+'\033[0;0m')

maxsnapshots = 5 #Numero de snapshots
spanspots = 0 
bol = True
print("-- Processo de deteccao iniciado --")
while(spanspots < maxsnapshots):

    if(bol == False):
        tempob = time.time()
        if(tempob - tempoa > 2):
            bol = True

    ret, frame = captura.read()
    cv2.imshow('Video', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, (board_w, board_h), None)

    if ret and (bol == True): #Bol controla o tempo de espera
        objpoints.append(objp)
        realobjpoints.append(realobjp)

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(frame, (board_w, board_h), corners, ret)
        cv2.imshow('Tabuleiro Detectado', frame)
        spanspots += 1
        print ('\033[1m'+'\033[31m'+'Proxima deteccao em 2 segundos'+'\033[0;0m')
        bol = False
        tempoa = time.time()

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        cv2.destroyAllWindows()
        exit()

print("\n")
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) 
file =  cv2.FileStorage("intrinsics.xml", cv2.FILE_STORAGE_WRITE)
file.write("intrinsics", mtx)
file.release()
file =  cv2.FileStorage("distortions.xml", cv2.FILE_STORAGE_WRITE)
file.write("distortions", dist)
file.release()

# ------------ Requisito 3 -----------------
#parametros intrinsecos da camera já encontrados
#mtx = Matrix Intrinseca
#dist = Coeficientes de distorcao

imgpoints2 = np.squeeze(imgpoints) #Ajustando imgpoints
rvecs2 = np.zeros((5,3,1))
tvecs2 = np.zeros((5,3,1))

for x in range(0,5): #Calibrando extrinsecos para cada snapshot
    retval2, rvecs2[x], tvecs2[x] = cv2.solvePnP(realobjpoints[x],imgpoints2[x],mtx,dist)
#Rvecs = Vetor de Rotacao
#Tvecs = Vetor de Translacao

distancias = calculaNormaTvecs(tvecs2) #calculando distancias
# ------------------------------------------

h, w = gray.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

cv2.namedWindow('raw')

while True:
    ret, frame = captura.read()

    cv2.setMouseCallback('raw',Requisito1)
    drawLines(frame,linhas)

    undist = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    cv2.imshow('undist', undist)
    cv2.imshow('raw', frame.astype(np.uint8))
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        cv2.destroyAllWindows()
        exit()

captura.release()
cv2.destroyAllWindows()