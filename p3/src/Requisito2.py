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

#Retifica a imagem stereo usando a matriz fundamental F
retBool ,rectmat1, rectmat2 = cv.stereoRectifyUncalibrated(p1fNew,p2fNew,F,(1200,1200))

#Ajeita a perspectiva das imagens para o caso da Matriz Fundamental
dst11 = cv.warpPerspective(img_left,rectmat1,(1200,1200))
dst22 = cv.warpPerspective(img_right,rectmat2,(1200,1200))

#Ajeita a perspectiva das imagens para o caso da Homografia
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
depth2 = depth2*254/np.amax(depth2)
depth2[filteredImg2<0] = 255

#Mostra os mapas de disparidade e profundidade encontrados
plt.imshow(filteredImg.astype(np.uint8),'jet')
plt.colorbar()
plt.clim(np.amax(filteredImg),0)
plt.title("Mapa de disparidade usando Matriz Fundamental")
plt.show()

plt.imshow(depth.astype(np.uint8),'jet')
plt.colorbar()
plt.clim(np.amax(depth),0)
plt.title("Mapa de profundidade usando Matriz Fundamental")
plt.show()

plt.imshow(filteredImg2.astype(np.uint8),'jet')
plt.colorbar()
plt.clim(np.amax(filteredImg2),0)
plt.title("Mapa de disparidade usando Homografia")
plt.show()

plt.imshow(depth2.astype(np.uint8),'jet')
plt.colorbar()
plt.clim(np.amax(depth2),0)
plt.title("Mapa de profundidade usando Homografia")
plt.show()

#Gera as saídas
print("Gerando arquivos de saída...")
cv.imwrite('../data/FurukawaPonce/disparidadeMF.pgm', filteredImg)
cv.imwrite('../data/FurukawaPonce/profundidadeMF.pgm', depth)
cv.imwrite('../data/FurukawaPonce/disparidadeH.pgm', filteredImg2)
cv.imwrite('../data/FurukawaPonce/profundidadeH.pgm', depth2)





