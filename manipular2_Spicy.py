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
    kernel_size = 3
    k = int ((kernel_size-1)/2)
    scipyImgFiltradaMediana = ndimage.median_filter(npImgOriginal, kernel_size)
    scipyImgFiltradaMedia= ndimage.gaussian_filter(npImgOriginal, 0)
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].imshow(npImgOriginal, cmap= 'gray')
    ax[0,0].set_title('Imagem Original')
    ax[1,0].imshow(scipyImgFiltradaMedia, cmap= 'gray')
    ax[1,0].set_title('Imagem Filtrada Media')
    ax[0,1].imshow(scipyImgFiltradaMediana, cmap= 'gray')
    ax[0,1].set_title('Imagem Filtrada Mediana')
    plt.show()



if __name__ == "__main__":
    main()