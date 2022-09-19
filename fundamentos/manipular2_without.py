import numpy as np
from PIL import Image
from PIL import ImageFilter
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import cv2 as cv

def main():
    img = Image.open('img/lena_gray_512_salt_pepper.tif')
    #img = Image.open('img/cameraman_salt_pepper.tif')
    #img = Image.open('img/house.tif')

    npImgOriginal = np.array(img)
    width, height = npImgOriginal.shape
    npImgFiltradaMedia = np.zeros((width,height))
    npImgFiltradaMediana = np.zeros((width,height))
    kernel_size = 3
    k = int ((kernel_size-1)/2)
    pllImgFiltradaMedia = img.filter(ImageFilter.BoxBlur(k))
    pllImgFiltradaMediana = img.filter(ImageFilter.MedianFilter(kernel_size))
    cvImgFiltradaMedia = cv.GaussianBlur(npImgOriginal, (kernel_size,kernel_size),0)
    cvImgFiltradaMediana = cv.medianBlur(npImgOriginal, kernel_size)
    scipyImgFiltradaMediana = ndimage.median_filter(npImgOriginal, kernel_size)
    
    for row in range(k, height-k):
        for col in range(k, width-k):
            npImgFiltradaMedia[row,col] = np.mean(npImgOriginal[row-k:row+k,col-k:col+k])
            npImgFiltradaMediana[row,col] = np.median(npImgOriginal[row-k:row+k,col-k:col+k])

    fig, ax = plt.subplots(nrows=2, ncols=4)
    ax[0,0].imshow(npImgOriginal, cmap= 'gray')
    ax[0,0].set_title('Imagem Original')
    ax[0,1].imshow(npImgFiltradaMedia, cmap= 'gray')
    ax[0,1].set_title('Imagem Filtrada Media Numpy')
    ax[0,2].imshow(pllImgFiltradaMedia, cmap = 'gray')
    ax[0,2].set_title('Imagem Filtrada Media Pillow')
    ax[0,3].imshow(cvImgFiltradaMedia, cmap = 'gray')
    ax[0,3].set_title('Imagem Filtrada Media OpenCv')
    ax[1,0].imshow(npImgFiltradaMediana, cmap= 'gray')
    ax[1,0].set_title('Imagem Filtrada Mediana Numpy')
    ax[1,1].imshow(pllImgFiltradaMediana, cmap= 'gray')
    ax[1,1].set_title('Imagem Filtrada Mediana Pillow')
    ax[1,2].imshow(cvImgFiltradaMediana, cmap= 'gray')
    ax[1,2].set_title('Imagem Filtrada Mediana OpenCv')
    ax[1,3].imshow(scipyImgFiltradaMediana, cmap= 'gray')
    ax[1,3].set_title('Imagem Filtrada Mediana Scipy')
    plt.show()



if __name__ == "__main__":
    main()