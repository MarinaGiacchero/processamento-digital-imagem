import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import sqrt,exp

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])*2 + (point1[1]-point2[1])*2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)*2)/(2(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)*2)/(2(D0**2))))
    return base

def main():

    
    img = cv2.imread("img/newspaper_shot_woman.tif", 0)

    imagem = Image.fromarray(img)

    original = np.fft.fft2(img)

    center = np.fft.fftshift(original)

    LowPassFilter = idealFilterLP(50,img.shape)

    LowPassCenter = center * idealFilterLP(50,img.shape)

    LowPass = np.fft.ifftshift(LowPassCenter)

    inverse_LowPass = np.fft.ifft2(LowPass)

    fig, ax = plt.subplots(nrows=2,ncols=3)
    ax[0,0].imshow(np.log(1+np.abs(imagem)),cmap='gray')
    ax[0,0].set_title('Original')
    ax[0,1].imshow(np.log(1+np.abs(original)),cmap='gray')
    ax[0,1].set_title('Espectral')
    ax[0,2].imshow(np.log(1+np.abs(LowPassFilter)),cmap='gray')
    ax[0,2].set_title('Passa Baixas')
    ax[1,0].imshow(np.log(1+np.abs(LowPassCenter)),cmap='gray')
    ax[1,0].set_title('Espectral Centralizada X Passa Baixa')
    ax[1,1].imshow(np.log(1+np.abs(LowPass)),cmap='gray')
    ax[1,1].set_title('Descentralizada')
    ax[1,2].imshow(np.log(1+np.abs(inverse_LowPass)),cmap='gray')
    ax[1,2].set_title('Processada')
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,2].set_xticks([])
    ax[0,2].set_yticks([])
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,2].set_xticks([])
    ax[1,2].set_yticks([])

    plt.show()



if __name__ == "__main__":
        main()