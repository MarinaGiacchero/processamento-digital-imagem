import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
import PIL
from scipy import ndimage, misc
from scipy.ndimage import affine_transform


def main():
    img = Image.open('img/lena_gray_512.tif') #para utilizar a imagem da mulher
    #img= Image.open('img/house.tif') #para utilizar a imagem da casa
    #img= Image.open('img/cameraman.tif') #para utilizar a imagem do cameraman

    img = np.array(img)
    width, height = img.shape
    (h, w)= img.shape[:2]
    center= (w/2, h/2)

    #imagem diminuida
    scale_percent = 50 
    width= int(img.shape[1] * scale_percent/100)
    height= int(img.shape[0] * scale_percent/100)
    dim= (width, height)
    imgdiminuida= cv.resize(img, dim, interpolation = cv.INTER_AREA)

    #imagem aumentada  
    scale_percent= 250 
    width= int(img.shape[1] * scale_percent / 100)
    height= int(img.shape[0] * scale_percent / 100)
    dim= (width, height)
    imgaumentada= cv.resize(img, dim, interpolation = cv.INTER_AREA)

    #rotar em 45
    Rotacao = cv.getRotationMatrix2D(center, 45, 1)
    img45 = cv.warpAffine(img, Rotacao, (w, h))

    #rotar em 90
    Rotacao = cv.getRotationMatrix2D(center, 90, 1)
    img90 = cv.warpAffine(img, Rotacao, (w, h))

    #rotar em 100
    Rotacao = cv.getRotationMatrix2D(center, 100, 1)
    img100 = cv.warpAffine(img, Rotacao, (w, h))

    #Translação 35x45
    deslocar = np.float32([[1, 0, 35], [0, 1, 45]])
    img35x45 = cv.warpAffine(img, deslocar, (h, w))
    #Translação de 100x20
    deslocar = np.float32([[1, 0, 100], [0, 1, 20]])
    imgAleatoria = cv.warpAffine(img, deslocar, (h, w))
 
    fig, ax= plt.subplots(nrows=2, ncols=4)
    ax[0,0].imshow(img, cmap= 'gray')
    ax[0,0].set_title('Imagem Original')
    ax[0,1].imshow(imgdiminuida, cmap= 'gray')
    ax[0,1].set_title('Imagem diminuida')
    ax[0,2].imshow(imgaumentada, cmap= 'gray')
    ax[0,2].set_title('Imagem aumentada')
    ax[0,3].imshow(img45, cmap= 'gray')
    ax[0,3].set_title('Rotação de 45°')
    ax[1,0].imshow(img90, cmap= 'gray')
    ax[1,0].set_title('Rotação de 90°')
    ax[1,1].imshow(img100, cmap='gray')
    ax[1,1].set_title('Rotação de 100°')
    ax[1,2].imshow(imgAleatoria, cmap= 'gray')
    ax[1,2].set_title('Translação aleatória')
    ax[1,3].imshow(img35x45, cmap= 'gray')
    ax[1,3].set_title('Translação de 35x45')

    plt.show()

if __name__ == "__main__":
    main()

