import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def GetFileData(arq):
    data = arq.readlines()

    #Cria estruturas de armazenamento
    extrinsic = np.zeros((3,3))
    pp = np.zeros(2)
    flenght = np.zeros(2)
    tc = np.zeros(3)

    #Le distancias focais
    conteudo = data[1]
    flenght[0] = float(conteudo[6:18])
    flenght[1] = float(conteudo[19:31])

    #Le centros opticos (principal points)
    conteudo = data[4]
    pp[0] = float(conteudo[6:17])
    pp[1] =  float(conteudo[18:29])

    #Le coeficiente skew
    conteudo = data[6]
    skew = float(conteudo[11:18])
    extrinsic = np.zeros((3,3))

    #Le matriz extrinseca da camera
    conteudo = data[11]
    extrinsic[0][0] = float(conteudo[6:16])
    extrinsic[0][1] = float(conteudo[17:29])
    extrinsic[0][2] = float(conteudo[30:42])
    conteudo = data[12]
    extrinsic[1][0] = float(conteudo[6:16])
    extrinsic[1][1] = float(conteudo[17:29])
    extrinsic[1][2] = float(conteudo[30:42])
    conteudo = data[13]
    extrinsic[2][0] = float(conteudo[6:16])
    extrinsic[2][1] = float(conteudo[17:29])
    extrinsic[2][2] = float(conteudo[30:42])

    #Le vetor de translacao
    conteudo = data[15]
    if(conteudo[3] != '8'):
        tc[0] = float(conteudo[7:18])
        tc[1] = float(conteudo[20:31])
        tc[2] = float(conteudo[33:44])
    else:
        tc[0] = float(conteudo[9:20])
        tc[1] = float(conteudo[22:33])
        tc[2] = float(conteudo[35:46])

    #Retorna dados obtidos
    return flenght, pp, extrinsic, tc


def ClickPosition(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        global cliques
        global points
        global img_copy
        #Para fazer 3 linhas precisa-se de 6 cliques
        if(cliques < 6):
            points[cliques] = (x,y)
            points = points.astype(int)
            #Dois pontos -> primeira linha formada
            if(cliques == 1):
                cv.line(img_copy, (points[0][0],points[0][1]),(points[1][0],points[1][1]), (0), 7)
                cv.imshow("image",img_copy.astype(np.uint8))
            #Quatro pontos -> segunda linha formada
            if(cliques == 3):
                cv.line(img_copy, (points[2][0],points[2][1]),(points[3][0],points[3][1]), (0), 7)
                cv.imshow("image",img_copy.astype(np.uint8))
            #Seis pontos -> terceira linha formada
            if(cliques == 5):
                cv.line(img_copy, (points[4][0],points[4][1]),(points[5][0],points[5][1]), (0), 7)
                cv.imshow("image",img_copy.astype(np.uint8))
            cliques += 1
            

#Excecao para casos de nao achar os arquivos .txt
try:
    arq_left = open('../data/FurukawaPonce/MorpheusL.txt','r')
    arq_right = open('../data/FurukawaPonce/MorpheusR.txt','r')
except(FileNotFoundError):
    print("Arquivos não encontrados, encerrando programa.")
    exit()

#Pega dados de cada camera
flenght_left, pp_left, extrinsinc_left, tc_left = GetFileData(arq_left)
flenght_right, pp_right, extrinsinc_right, tc_right = GetFileData(arq_right)

img_left = cv.imread("../data/FurukawaPonce/MorpheusL.jpg",0)
img_right = cv.imread("../data/FurukawaPonce/MorpheusR.jpg",0)

# Inicializando detector SIFT
sift = cv.xfeatures2d.SIFT_create()

# Achando os pontos chaves e descritores
kp1, des1 = sift.detectAndCompute(img_left,None)
kp2, des2 = sift.detectAndCompute(img_right,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1, des2, 2)

#Teste de ratio para achar melhores matches 
pts1 = []
pts2 = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

#Acha a matriz fundamental
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_RANSAC)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#Acha a homografia
h, status = cv.findHomography(pts1, pts2,cv.RANSAC)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))

#Retifica a imagem stereo
retBool ,rectmat1, rectmat2 = cv.stereoRectifyUncalibrated(p1fNew,p2fNew,F,(1200,1200))

#Ajeita a perspectiva das imagens
dst11 = cv.warpPerspective(img_left,rectmat1,(1200,1200))
dst22 = cv.warpPerspective(img_right,rectmat2,(1200,1200))

#Ajeita a perspectiva das imagens para o caso da Matriz Fundamental
dst22l = cv.warpPerspective(img_left,h,(1200,1200))

stereo2 = cv.StereoSGBM_create(minDisparity=0,numDisparities = 160, blockSize = 5, P1=8*3*5**2, 
                                P2=32*3*5**2,disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=0,speckleRange=2,
                                preFilterCap=63,mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

stereo2_matcher = cv.ximgproc.createRightMatcher(stereo2)
# FILTER Parameters, aplica filtros para diminuir o ruído
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo2)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#Computa o mapa de disparidade usando Matriz Fundamental
displ = stereo2.compute(dst11, dst22)  
dispr = stereo2_matcher.compute(dst22, dst11)  
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, dst11, None, dispr)
#Normaliza o mapa de disparidade
cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)

#Calculando mapa de disparidade para caso de Homografia
stereo = cv.StereoSGBM_create(minDisparity=0,numDisparities = 160, blockSize = 5, P1=8*3*5**2, 
                            P2=32*3*5**2,disp12MaxDiff=1,uniquenessRatio=15,speckleWindowSize=0,speckleRange=2,
                            preFilterCap=63,mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

stereo_matcher = cv.ximgproc.createRightMatcher(stereo)
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
displ2 = stereo.compute(dst22l, img_right)  # .astype(np.float32)/16
dispr2 = stereo_matcher.compute(img_right, dst22l)  # .astype(np.float32)/16
displ2 = np.int16(displ2)
dispr2 = np.int16(dispr2)
filteredImg2 = wls_filter.filter(displ2, dst22l, None, dispr)
cv.normalize(src=filteredImg2, dst=filteredImg2, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)

#Acha doffs e baseline
doffs = np.linalg.norm(pp_left-pp_right)
baseline = np.linalg.norm(tc_left-tc_right)

#Calcula o mapa de profundidade para o caso da matriz fundamental
depth = (baseline*flenght_left[0])/(filteredImg + doffs)
depth = depth*254/np.amax(depth)
depth[filteredImg<0] = 255

#Calcula o mapa de profundidade para o caso da homografia
depth2 = (baseline*flenght_left[0])/(filteredImg2 + doffs)
depth2 = depth*254/np.amax(depth)
depth2[filteredImg2<0] = 255

#Preparando imagem que será mostrada
crop_img = dst11[0:1200, 600:1200]
crop_img2 = dst22[0:1200, 0:600]
stereo_img = np.concatenate((crop_img2, crop_img), axis=1)
img_copy = stereo_img
cliques = 0

#Loop das linhas
cv.namedWindow('image')
cv.imshow("image",img_copy.astype(np.uint8))
points = np.zeros((6,2))
while(1):
    cv.setMouseCallback("image",ClickPosition)
    if(cliques == 6): #3 linhas apenas, 6 pontos
        break
    if cv.waitKey(250) & 0xFF == ord('q'): #Fecha imagem a cada 25 ms
        break   

#Matriz de transformacao perspectiva
q = np.zeros((4,4))
q[0][0] = 1
q[1][1] = 1
q[3][2] = -1/baseline
q[0][3] = -1*pp_left[0]
q[1][3] = -1*pp_left[1]
q[2][3] = flenght_left[0]
q[3][3] = (pp_left[0] - pp_right[0])/baseline

#Convertendo para pontos 3D
img_3d = cv.reprojectImageTo3D(filteredImg, q)

#Medidas das 3 linhas feitas pelo usuario
medidas = np.zeros(3)
medidas[0] = np.linalg.norm(img_3d[points[0][0],points[0][1]] - img_3d[points[1][0],points[1][1]])
medidas[1] = np.linalg.norm(img_3d[points[2][0],points[2][1]] - img_3d[points[3][0],points[3][1]])
medidas[2] = np.linalg.norm(img_3d[points[4][0],points[4][1]] - img_3d[points[5][0],points[5][1]])

#Ordena pelo tamanho
medidas = np.sort(medidas)
altura = medidas[2] #Altura sempre vai ser a maior medida nesse caso
largura = medidas[1] 
profundidade = medidas[0] #Profundidade sempre vai ser a menor medida
print("-- Dados do Sofá (em mm)--") #Pelo dados obtidos inferiu-se que as medidas estao em decimos de milimetros
print("Altura: {1:.1f}\nLargura: {0:.1f}\nProfundidade: {2:.1f}".format(largura/10,altura/10,profundidade/10))
print("Volume da caixa: {0:.1f} mm^3".format(altura*largura*profundidade/1000))
