#bibliotecas
from statistics import median
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    #img= Image.open('img/lena_gray_512.tif')
    #img= Image.open('img/cameraman.tif')
    img= Image.open('img/biel.png')

    # mean  
    mean = np.array((
        [0.1111, 0.1111, 0.1111],
        [0.1111, 0.1111, 0.1111],
        [0.1111, 0.1111, 0.1111]), dtype="float")

    # gauss
    gauss = np.array((
        [0.0625, 0.125, 0.0625], 
        [0.1250, 0.250, 0.1250],
        [0.0625, 0.125, 0.0625]), dtype="float")

    # laplaciano
    laplacian = np.array((
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]), dtype="int")

    # sobelX
    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), dtype="int")
   

    # sobelY
    sobelY = np.array((
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]), dtype="int")  

    npImgOriginal= np.array(img)
    width, height= npImgOriginal.shape
    imgMedia = np.zeros(shape=(width,height))
    imgGauss= np.zeros(shape=(width,height))
    imglaplacian= np.zeros(shape=(width,height))
    imgX= np.zeros(shape=(width,height))
    imgY= np.zeros(shape=(width,height))
    kernel= mean.shape[0]

    for row in range(height-2):
        for col in range(width-2):
            imagem= npImgOriginal[row:row+kernel, col:col+kernel]
            imgMedia[row,col] = np.sum(np.multiply(imagem, mean))
            imgGauss[row,col] = np.sum(np.multiply(imagem, gauss))
            imglaplacian[row,col] = np.sum(np.multiply(imagem, laplacian))
            imgX[row,col] = np.sum(np.multiply(imagem, sobelX))
            imgY[row,col] = np.sum(np.multiply(imagem, sobelY))
    
    # grad
    grad= imgX+imgY

    # Laplaciano + Original
    originalLaplace= img+imglaplacian

    fig, ax = plt.subplots(nrows=2, ncols=4)
    ax[0,0].imshow(img, cmap= 'gray')
    ax[0,0].set_title('Imagem Original')
    ax[0,1].imshow(imgMedia, cmap= 'gray')
    ax[0,1].set_title('Máscara Media')
    ax[1,0].imshow(imgGauss, cmap= 'gray')
    ax[1,0].set_title('Máscara Gaussiana')
    ax[1,1].imshow(imglaplacian, cmap= 'gray')
    ax[1,1].set_title('Máscara Laplaciano')
    ax[0,2].imshow(imgX, cmap= 'gray')
    ax[0,2].set_title('Máscara Sobel X')
    ax[1,2].imshow(imgY, cmap= 'gray')
    ax[1,2].set_title('Máscara Sobel Y')
    ax[0,3].imshow(grad, cmap= 'gray')
    ax[0,3].set_title('Máscara Gradiente')
    ax[1,3].imshow(originalLaplace, cmap= 'gray')
    ax[1,3].set_title('Máscara Laplaciano + Original')  
    plt.show()

if __name__ == "__main__":
    main()