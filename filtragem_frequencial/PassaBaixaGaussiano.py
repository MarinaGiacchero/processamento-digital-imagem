import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import sqrt,exp

    # Calculando a distância entre os dois pontos
def distancia(primeiroPonto, segundoPonto):
    return sqrt((primeiroPonto[0]-segundoPonto[0])**2 + (primeiroPonto[1]-segundoPonto[1])**2)

def main():
    # Fazendo leitura das imagens
    img = cv2.imread("img/newspaper_shot_woman.tif", 0)
    #img = cv2.imread("img/car.tif", 0)
#    img = cv2.imread("img/len_periodic_noise.png", 0)
   # img = cv2.imread("img/periodic_noise.png", 0)
    imagem = Image.fromarray(img)
    espectral = np.fft.fft2(img)
    Centro = np.fft.fftshift(espectral)

    # Aplicando o Gaussiano
    base = np.zeros(img.shape[:2])
    rows, cols = img.shape[:2]
    centro = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distancia((y,x),centro)**2)/(2*(50**2))))

    PassaBaixaFilter = base
    PassaBaixaCentro = Centro * base
    PassaBaixa = np.fft.ifftshift(PassaBaixaCentro)
    inversoPassaBaixa = np.fft.ifft2(PassaBaixa)

    fig, ax = plt.subplots(nrows=2,ncols=3)
    ax[0,0].imshow(np.log(1+np.abs(imagem)),cmap='gray')
    ax[0,0].set_title('Original')
    ax[0,1].imshow(np.log(1+np.abs(espectral)),cmap='gray')
    ax[0,1].set_title('Espectral')
    ax[0,2].imshow(np.log(1+np.abs(PassaBaixaFilter)),cmap='gray')
    ax[0,2].set_title('Passa-Baixas BW')
    ax[1,0].imshow(np.log(1+np.abs(PassaBaixaCentro)),cmap='gray')
    ax[1,0].set_title('Espectral Centralizada c/ Passa-Baixa BW')
    ax[1,1].imshow(np.log(1+np.abs(PassaBaixa)),cmap='gray')
    ax[1,1].set_title('Descentralizada')
    ax[1,2].imshow(np.log(1+np.abs(inversoPassaBaixa)),cmap='gray')
    ax[1,2].set_title('Resultado final')

    plt.show()

if __name__ == "__main__":
        main()

