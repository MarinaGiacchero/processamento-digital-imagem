#bibliotecas
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
from PIL import Image
#fazendo a leitura da imagem
img = cv.imread('img/Fig308.tif',0)  #primeira imagem
#img = Image.open('img/enhance-me.gif') #segunda imagem
#img= np.array(img)
#criando uma lista
lista=[]
#fazendo um laço com número de bits da imagem criada na lista
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        lista.append(np.binary_repr(img[i][j], width=8))

#separando por cada bit
imgOitoBits = (np.array([int(i[0]) for i in lista], dtype= np.uint8)*128).reshape(img.shape[0], img.shape[1])
imgSeteBits = (np.array([int(i[1]) for i in lista], dtype= np.uint8)*64).reshape(img.shape[0], img.shape[1])
imgSeisBits = (np.array([int(i[2]) for i in lista], dtype= np.uint8)*32).reshape(img.shape[0], img.shape[1])
imgCincoBits = (np.array([int(i[3]) for i in lista], dtype= np.uint8)*16).reshape(img.shape[0], img.shape[1])
imgQuatroBits = (np.array([int(i[4]) for i in lista], dtype= np.uint8)*8).reshape(img.shape[0], img.shape[1])
imgTresBits = (np.array([int(i[5]) for i in lista], dtype= np.uint8)*4).reshape(img.shape[0], img.shape[1])
imgDoisBits = (np.array([int(i[6]) for i in lista], dtype= np.uint8)*2).reshape(img.shape[0], img.shape[1])
imgUmBits = (np.array([int(i[7]) for i in lista], dtype= np.uint8)*1).reshape(img.shape[0], img.shape[1])

#concatenando em duas partes
primeiro= cv.hconcat([imgOitoBits, imgSeteBits, imgSeisBits, imgCincoBits])
segundo= cv.hconcat([imgQuatroBits, imgTresBits, imgDoisBits, imgUmBits])
#concatenando as duas partes
imgPlano= cv.vconcat([primeiro, segundo])
#apresentar na tela
fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(img,cmap='gray')
ax[0].set_title('Imagem original')
ax[1].imshow(imgPlano,cmap='gray')
ax[1].set_title('Cada plano de bits')
plt.show()