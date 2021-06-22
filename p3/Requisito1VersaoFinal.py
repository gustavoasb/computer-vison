import numpy as np
import cv2
from matplotlib import pyplot as plt


######################################################################################################################################################

print ('\033[1m'+'\033[35m'+'+----------------------------------------+'+'\033[0;0m')
print ('\033[1m'+'\033[37m'+'|           Tamanho da Janela'+'\033[0;0m')
print ('\033[1m'+'\033[37m'+'+----------------------------------------+'+'\033[0;0m')
print ('\033[1m'+'\033[37m'+'|  Valores ímpares no intervalo [5, 255]'+'\033[0;0m')
print ('\033[1m'+'\033[35m'+'+----------------------------------------+'+'\033[0;0m')

winsize = int(input('\033[1m'+'\033[36m'+'Tamanho da Janela para o SAD: '+'\033[0;0m'))

if winsize % 2 == 0:
    print ('\033[1m'+'\033[31m'+'\nO valor dado para o Tamanho da Danela deve ser ímpar'+'\033[0;0m')
    raise SystemExit
elif winsize < 5 or winsize > 255:
    print ('\033[1m'+'\033[31m'+'\nO valor dado para o Tamanho da Janela deve estar no intervalo [5, 255]'+'\033[0;0m')
    raise SystemExit

print("\n")

######################################################################################################################################################

print ('\033[1m'+'\033[35m'+'+------------------------------+'+'\033[0;0m')
print ('\033[1m'+'\033[37m'+'|  Imagens     Valor Referente'+'\033[0;0m')
print ('\033[1m'+'\033[37m'+'+------------------------------+'+'\033[0;0m')
print ('\033[1m'+'\033[37m'+'|   Moto              1'+'\033[0;0m')
print ('\033[1m'+'\033[37m'+'|  Planta             2'+'\033[0;0m')
print ('\033[1m'+'\033[35m'+'+------------------------------+'+'\033[0;0m')

imagem = int(input('\033[1m'+'\033[36m'+'Digite o Valor Referente a imagem desejada: '+'\033[0;0m'))
    
if imagem == 1:
    arq = open('Imagens/Moto/calib.txt', 'r')
    imgL = cv2.imread('Imagens/Moto/im1.png',0)
    imgR = cv2.imread('Imagens/Moto/im0.png',0)
elif imagem == 2:
    arq = open('Imagens/Plant/calib.txt', 'r')
    imgL = cv2.imread('Imagens/Plant/im1.png',0)
    imgR = cv2.imread('Imagens/Plant/im0.png',0)
else:
    print ('\033[1m'+'\033[31m'+'\nO valor dado não é referente a nenhuma imagem'+'\033[0;0m')
    raise SystemExit

print('\033[1m'+'\033[37m'+'\nImagens carregadas\n'+'\033[0;0m')

######################################################################################################################################################

texto = arq.readlines()
conteudo = texto[0]
focallength = float(conteudo[6:14])
#print(focallenght)
conteudo = texto[2]
doffs = float(conteudo[6:])
#print(doffs)
conteudo = texto[3]
baseline = float(conteudo[9:])
#print(baseline)
conteudo = texto[4]
widht = int(conteudo[6:])
#print(widht)
conteudo = texto[6]
ndisp = int(conteudo[6:])
#print(ndisp)
arq.close()

######################################################################################################################################################

#cv2.error: numDisparities must be positive and divisble by 16 in function 'compute'
#cv2.error: SADWindowSize must be odd, be within 5..255 and be not larger than image width or height in function 'compute'
if ndisp / 16 != 0: #NumDisparities so pode receber um valor multiplo de 16, entao o valor e arredondado para o multiplo de 16 mais proximo
    x = ndisp % 16
    if x > 16 - x:
        ndisp = ndisp + 16 - x
    else:
        ndisp = ndisp - x

######################################################################################################################################################

#Disparity range is tuned for image pair
window_size = 3
min_disp = 0
num_disp = ndisp - min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = winsize,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

######################################################################################################################################################

print('\033[1m'+'\033[37m'+'Computando disparidade...\n'+'\033[0;0m')

disparity = stereo.compute(imgR, imgL).astype(np.float32) / 16.0

print ('\033[1m'+'\033[36m'+'Tecle "q" para fechar a janela e computar a profundidade\n'+'\033[0;0m')

plt.imshow(disparity, 'gray')
plt.show()

######################################################################################################################################################

print('\033[1m'+'\033[37m'+'Computando profundidade...\n'+'\033[0;0m')

#To convert from the floating-point disparity value d [pixels] in the .pfm file to depth Z [mm] the following equation can be used:
#Z = baseline * f / (d + doffs)
depth = (baseline*focallength)/(disparity + doffs)

depth = depth*254/np.amax(depth)
depth[disparity<0] = 255

print ('\033[1m'+'\033[36m'+'Tecle "q" para fechar a janela e gerar as imagens de saída\n'+'\033[0;0m')

plt.imshow(depth, 'gray')
plt.show()

cv2.destroyAllWindows()

######################################################################################################################################################

if imagem == 1:
    cv2.imwrite('Imagens/Moto/disparidade.pgm', disparity)
    cv2.imwrite('Imagens/Moto/profundidade.pgm', depth)
else:
    cv2.imwrite('Imagens/Plant/disparidade.pgm', disparity)
    cv2.imwrite('Imagens/Plant/profundidade.pgm', depth)

print ('\033[1m'+'\033[37m'+'As imagens disparidade.pgm e profundidade.pgm foram geradas'+'\033[0;0m')

######################################################################################################################################################