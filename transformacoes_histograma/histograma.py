#bibliotecas
from ctypes import sizeof
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image

#fazendo a leitura da imagem
img = cv.imread('img/Fig308.tif')  #primeira imagem
#img = Image.open('img/enhance-me.gif') #segunda imagem
#img= np.array(img) #alterar o tipo caso segunda imagem

#alterando a cor
imgH = cv.cvtColor(img,cv.COLOR_BGR2YUV)
#utilizando a função de equalização de histograma do OpenCv
imgH[:,:,0] = cv.equalizeHist(imgH[:,:,0])
#voltando a cor
imgHisto = cv.cvtColor(imgH, cv.COLOR_YUV2BGR)
 
#apresentar na tela
fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(img,cmap='gray')
ax[0].set_title('Imagem original')
ax[1].imshow(imgHisto,cmap='gray')
ax[1].set_title('Equalização de Histograma')
plt.show()