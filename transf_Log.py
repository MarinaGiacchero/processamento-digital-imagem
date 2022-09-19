import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
#img = Image.open('img/Fig308.tif') 
img = Image.open('img/enhance-me.gif') 
imgAnt = np.array(img)  
c = 1 
y= 0.1
imgGama = c * (np.power(imgAnt, y)) 
y= 1.5
imgGama2 = c * (np.power(imgAnt, y)) 
y=3.0
imgGama3 = c * (np.power(imgAnt, y)) 
#apresentar na tela
fig, ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].imshow(img,cmap='gray')
ax[0,0].set_title('Imagem original')  
ax[0,1].imshow(imgGama,cmap='gray')
ax[0,1].set_title('Imagem transformada com y=0.1') 
ax[1,0].imshow(imgGama2,cmap='gray')
ax[1,0].set_title('Imagem transformada com y= 1.5') 
ax[1,1].imshow(imgGama3,cmap='gray')
ax[1,1].set_title('Imagem transformada com y= 3.0') 
plt.show()