import cv2
import numpy as np
import time

#Calcula a norma do vetor de translaçao
def calculaNormaTvecs(tvecs):
    distance = np.zeros(5) 
    for x in range(0,5): #5 snapshots
        distance[x] = ((tvecs[x][0]*tvecs[x][0])+(tvecs[x][1]*tvecs[x][1])+(tvecs[x][2]*tvecs[x][2]))**(1/2)
        print("Distância {}: {:.2f} mm".format(x,distance[x]))
    print("Distância media = {:.2f} mm".format(np.mean(distance)))
    print("Desvio padrao = {:.2f} mm".format(np.std(distance)))
    return distance


#Lendo matrix intrinseca da camera
fs_i = cv2.FileStorage("intrinsics.xml", cv2.FILE_STORAGE_READ)
fread_i = fs_i.getNode("intrinsics")
cameraMatrix = fread_i.mat()

#Lendo parametros de distorcao
fs_d = cv2.FileStorage("distortions.xml", cv2.FILE_STORAGE_READ)
fread_d = fs_i.getNode("distortions")
distsCoeffs = fread_d.mat()

#Release nos arquivos lidos
fs_d.release()
fs_i.release()

# Linhas e colunas do Tabuleiro de Xadrez do Pattern
board_w = 8
board_h = 6

# Linhas e colunas do Tabuleiro de Xadrez do Pattern (tamanho real) em CM
board_rw = 0.0
board_rh = 0.0
board_rw = input("Insira o tamanho da largura de um retangulo no chessboard (em MM). Recomendado: 28\n")
board_rh = input("Insira o tamanho da altura de um retangulo no chessboard(em MM). Recomendado: 28\n")
if((board_rw.isnumeric() == False) or (board_rh.isnumeric() == False)): #Caso para input errado
    print("Tamanho nao definido, encerrando programa")
    exit()
board_rw = float(board_rw) #é necessasrio ser float para
board_rh = float(board_rh) #evitar conflito na conta

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_h*board_w, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# prepare object points, like (0,0,0), (28,0,0), (56,0,0) ....,(168,140,0)
# porém agora leva em consideração medidas reais escolhidas pelo usuario
realobjp = np.zeros((board_h*board_w, 3), np.float32)
realobjp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
# prepare object points using MM given
for x in range(0,48):
    realobjp[x][0] = realobjp[x][0]*board_rw
    realobjp[x][1] = realobjp[x][1]*board_rh

# Arrays to store object points and image points from all the images.
realobjpoints = []  # 3d point in real world space
objpoints = []  # 3d point in unit world space

#Detecta o chessboard e ja executa o calculo dos extrinsicos
def detectAndCalcuteExtrinsics(cameraMatrix,distsCoeffs):
    imgpoints = [] # 2d points in image plane.
    global realobjp
    global realobjpoints
    global objp
    global objpoints
    captura = cv2.VideoCapture(0)
    maxsnapshots = 5 #Numero snapshots
    spanspots = 0
    bol = True #Booleana de tempo
    print("Processo de deteccao iniciado")
    while(spanspots < maxsnapshots): 

        if(bol == False):
            tempob = time.time()
            if(tempob - tempoa > 2): #2 segundos entre capturas
                bol = True

        ret, frame = captura.read()
        cv2.imshow('Video', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (board_w, board_h), None)

        if ret and (bol == True): #se bol é true já passou 2 segundos
            objpoints.append(objp) #vetor de pontos do objeto (unit)
            realobjpoints.append(realobjp) #vetor de pontos do objeto (MM)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, (board_w, board_h), corners, ret)
            cv2.imshow('Tabuleiro Detectado', frame)
            spanspots += 1
            print ("Proxima deteccao em 2 segundos")
            bol = False
            tempoa = time.time()
            

        k = cv2.waitKey(25) & 0xFF
        if k == ord("q"):
            cv2.destroyAllWindows()
            captura.release()
            exit()

    print("\n")
    cv2.destroyAllWindows()
    imgpoints2 = np.squeeze(imgpoints) #Ajustando imgpoints
   
    rvecs = np.zeros((5,3,1))
    tvecs = np.zeros((5,3,1))
    for x in range(0,5): #Calibrando extrinsecos para cada snapshot
        retval, rvecs[x], tvecs[x] = cv2.solvePnP(realobjpoints[x],imgpoints2[x],cameraMatrix,distsCoeffs)
    distancias = calculaNormaTvecs(tvecs)
    print("\n")
    del imgpoints #Liberando imgpoints para a proxima chamada de funcao
    return rvecs,tvecs

print("Primeira distancia")
rvecs1, tvecs1 = detectAndCalcuteExtrinsics(cameraMatrix,distsCoeffs)
print("Segunda distancia em 5 segundos")
time.sleep(5) #5 segundos antes de executar proximo bloco
rvecs2, tvecs2 = detectAndCalcuteExtrinsics(cameraMatrix,distsCoeffs)
print("Terceira distancia em 5 segundos")
time.sleep(5)
rvecs3, tvecs3 = detectAndCalcuteExtrinsics(cameraMatrix,distsCoeffs)