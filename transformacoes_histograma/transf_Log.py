import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#img = cv.imread('img/Fig308.tif')
img = Image.open('img/enhance-me.gif') #segunda imagem
img= np.array(img)
c = 10 / np.log(1 + np.max(img))
log_img = c * (np.log(img + 1))
log_img = np.array(log_img, dtype = np.uint8)  
#apresentar na tela
fig, ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].imshow(img,cmap='gray')
ax[0,0].set_title('Imagem original')  
ax[0,1].imshow(log_img,cmap='gray')
ax[0,1].set_title('Imagem transformada')
 
plt.show()
