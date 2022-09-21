from scipy import signal
from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#img= Image.open('img/lena_gray_512.tif')
#img= Image.open('img/cameraman.tif')
img= Image.open('img/biel.png')
imgOriginal= np.array(img)
# mean  
mean = np.array((
    [0.1111, 0.1111, 0.1111],
    [0.1111, 0.1111, 0.1111],
    [0.1111, 0.1111, 0.1111]), dtype="float")
media = cv.filter2D(imgOriginal,-1, mean)

# gauss
gauss = np.array((
    [0.0625, 0.125, 0.0625], 
    [0.1250, 0.250, 0.1250],
    [0.0625, 0.125, 0.0625]), dtype="float")
gaussiano = cv.filter2D(imgOriginal, -1, gauss)

# laplaciano
laplacian = np.array((
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]), dtype="int")
laplace = cv.filter2D(imgOriginal, -1, laplacian)

# sobelX
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
x= cv.filter2D(imgOriginal, -1, sobelX)

# sobelY
sobelY = np.array((
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]), dtype="int")  
y= cv.filter2D(imgOriginal, -1, sobelY)

# grad
grad= x+y

# Laplaciano + Original
originalLaplace= imgOriginal+laplace

fig, ax = plt.subplots(nrows=2, ncols=4)
ax[0,0].imshow(imgOriginal, cmap= 'gray')
ax[0,0].set_title('Imagem Original')
ax[0,1].imshow(media, cmap= 'gray')
ax[0,1].set_title('Máscara Media')
ax[1,0].imshow(gaussiano, cmap= 'gray')
ax[1,0].set_title('Máscara Gaussiana')
ax[1,1].imshow(laplace, cmap= 'gray')
ax[1,1].set_title('Máscara Laplaciano')
ax[0,2].imshow(x, cmap= 'gray')
ax[0,2].set_title('Máscara Sobel X')
ax[1,2].imshow(y, cmap= 'gray')
ax[1,2].set_title('Máscara Sobel Y')
ax[0,3].imshow(grad, cmap= 'gray')
ax[0,3].set_title('Máscara Gradiente')
ax[1,3].imshow(originalLaplace, cmap= 'gray')
ax[1,3].set_title('Máscara Laplaciano + Original')

plt.show()
