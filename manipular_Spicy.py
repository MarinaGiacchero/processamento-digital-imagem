import cv2
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

    npImgOriginal = np.array(img)
    width, height = npImgOriginal.shape
    img45 = ndimage.rotate(img, 45, reshape=False)
    img90 = ndimage.rotate(img, 90, reshape=False)
    img100 = ndimage.rotate(img, 100, reshape=False)
    imgAumentada=ndimage.zoom(img, 2.5)
    imgDiminuida=ndimage.zoom(img, 0.66)

    centre = np.identity(3)
    centre[:2,2] = -width/2.,-height/2.
    theta = np.deg2rad(45)
    rotate = np.identity(3)
    rotate[:2,:2] = [np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]
    translate = np.identity(3)
    translate[:2,2] = 300,500
    affine = np.dot(translate,np.dot(rotate,centre))
    rmat = affine[:2,:2]
    offset = affine[:2,2]
    teste=affine_transform(img,rmat,offset=offset,output_shape=(512,512))

    fig, ax= plt.subplots(nrows=2, ncols=4)
    ax[0,0].imshow(img, cmap= 'gray')
    ax[0,0].set_title('Imagem Original')
    ax[0,1].imshow(imgAumentada, cmap= 'gray')
    ax[0,1].set_title('Imagem aumentada')
    ax[0,2].imshow(imgDiminuida, cmap= 'gray')
    ax[0,2].set_title('Imagem diminuída')
    ax[0,3].imshow(img45, cmap= 'gray')
    ax[0,3].set_title('Rotação de 45°')
    ax[1,0].imshow(img90, cmap= 'gray')
    ax[1,0].set_title('Rotação de 90°')
    ax[1,1].imshow(img100, cmap='gray')
    ax[1,1].set_title('Rotação de 100°')

    plt.show()

if __name__ == "__main__":
    main()

