import requisito1
import requisito2
import requisito3
import requisito4
import os
import cv2

def printOptions():
    print('(1) Show pixel coordenates and color')
    print('(2) Color similitary (picture)')
    print('(3) Color similitary (video)')
    print('(4) Color similitary (webcam)')
    print('Type any other key to quit')

printOptions()
while(1):
    bol = 0
    key = input('Select one of the options above: ')
    os.system('cls' if os.name == 'nt' else 'clear')
    if(key == '1'):
        requisito1.main()
        print('\n')
        printOptions()
        bol = 1
    if(key == '2'):
        requisito2.main()
        print('\n')
        printOptions()
        bol = 1
    if(key == '3'):
        requisito3.main()
        print('\n')
        printOptions()
        bol = 1
    if(key == '4'):
        requisito4.main()
        print('\n')
        printOptions()
        bol = 1
    elif(bol == 0):
        break

        