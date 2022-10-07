import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import sqrt,exp

def distancia(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def main():
    
    img = cv2.imread("img/newspaper_shot_woman.tif", 0)
    imagem = Image.fromarray(img)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    
    # Filtro ideal passa baixa
    base = np.zeros(img.shape[:2])
    rows, cols = img.shape[:2]
    meio = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distancia((y,x),meio) < 100:
                base[y,x] = 1

    PassaBaixa = base

    #Filtro ideal passa alta
    base = np.ones(img.shape[:2])
    rows, cols = img.shape[:2]
    meio = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distancia((y,x),meio) < 50:
                base[y,x] = 0
   
    PassaAlta = base

    PassaBanda = PassaAlta - PassaBaixa
    PassaCentro = center * PassaBanda

    PassaBaixaFiltro = np.fft.ifftshift(PassaCentro)

    inversoPassaBaixaFiltro = np.fft.ifft2(PassaBaixaFiltro)

    fig, ax = plt.subplots(nrows=2,ncols=3)
    ax[0,0].imshow(np.log(1+np.abs(imagem)),cmap='gray')
    ax[0,0].set_title('Imagem original')
    ax[0,1].imshow(np.log(1+np.abs(original)),cmap='gray')
    ax[0,1].set_title('Espectral')
    ax[0,2].imshow(np.log(1+np.abs(PassaBanda)),cmap='gray')
    ax[0,2].set_title('Passa-Banda')
    ax[1,0].imshow(np.log(1+np.abs(PassaCentro)),cmap='gray')
    ax[1,0].set_title('Espectral Centralizada X Passa-Banda')
    ax[1,1].imshow(np.log(1+np.abs(PassaBaixaFiltro)),cmap='gray')
    ax[1,1].set_title('Descentralizada')
    ax[1,2].imshow(np.log(1+np.abs(inversoPassaBaixaFiltro)),cmap='gray')
    ax[1,2].set_title('Resultado Final')

    plt.show()

if __name__ == "__main__":
        main()

