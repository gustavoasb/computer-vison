import numpy as np
import cv2
from matplotlib import pyplot as plt


winsize = int(input('Tamanho da janela para o SAD: '))

if int(winsize) % 2 == 0:
    print("O valor dado para o tamanho da janela deve ser impar")
    raise SystemExit
elif int(winsize) < 5 or int(winsize) > 255:
    print("O valor dado para o tamanho da janela deve estar no intervalo [5, 255]")
    raise SystemExit

imagem = int(input('Para escolher as imagens, digite 1 para moto ou 2 para plant: '))
if imagem == 1:
    arq = open('Imagens/Moto/calib.txt', 'r')
elif imagem == 2:
    arq = open('Imagens/Plant/calib.txt', 'r')
else:
    print("A imagem escolhida e invalida")
    raise SystemExit

texto = arq.readlines()
conteudo = texto[0]
f = float(conteudo[6:14])
#print(f)
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

imgL = cv2.imread('im1.png',0)
imgR = cv2.imread('im0.png',0)

#cv2.error: numDisparities must be positive and divisble by 16 in function 'compute'
#cv2.error: SADWindowSize must be odd, be within 5..255 and be not larger than image width or height in function 'compute'
#Substituir 272 por ndisp
stereo = cv2.StereoBM_create(numDisparities = 272, blockSize = winsize)
disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity, 'gray')
plt.show()

cv2.waitKey(0)

if imagem == 1:
    cv2.imwrite('Imagens/Moto/disparidade.pgm', disparity)
else:
    cv2.imwrite('Imagens/Plant/disparidade.pgm', disparity)
