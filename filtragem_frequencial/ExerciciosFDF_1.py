import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # a- Criando a imagem
    img = np.ones((512, 512)).astype(int)
    img[200:315,200:315] = 255

    imagem = Image.fromarray(img)

    #b- transformada de fourier (amplitude)
    img_fourier1 = np.fft.fft2(img)

    #c- transformada de fourier (fases)
    img_ftt = np.fft.fftshift(np.fft.fft2(img))
    img_fourier2= np.angle(img_ftt)

    #d- espectro de fourier contralizado
    img_fourier3 = np.fft.fftshift(img_fourier1)

    #e- Rotacionar a imagem e aplicar passos B e D
    img40= imagem.rotate(40)
    img40_fourier1 = np.fft.fft2(img40)
    img40_ftt = np.fft.fftshift(np.fft.fft2(img40))
    img40_fourier2= np.angle(img40_ftt)
    img40_fourier3 = np.fft.fftshift(img40_fourier1)

    #f- translação nos eixos x e y e aplicar os passo b-d
    img_deslocada=imagem.rotate(0, translate=(50,200))
    img_deslocada_fourier1 = np.fft.fft2(img_deslocada)
    img_deslocada_ftt = np.fft.fftshift(np.fft.fft2(img_deslocada))
    img_deslocada_fourier2= np.angle(img_deslocada_ftt)
    img_deslocada_fourier3 = np.fft.fftshift(img_deslocada_fourier1)

    fig, ax = plt.subplots(nrows=3,ncols=4)
    ax[0,0].imshow(np.log(1+np.abs(img)),cmap='gray')
    ax[0,0].set_title('Original')
    ax[0,1].imshow(np.log(1+np.abs(img_fourier1)),cmap='gray')
    ax[0,1].set_title('Amplitude')
    ax[0,2].imshow(np.log(1+np.abs(img_fourier2)),cmap='gray')
    ax[0,2].set_title('Fases')
    ax[0,3].imshow(np.log(1+np.abs(img_fourier3)),cmap='gray')
    ax[0,3].set_title('Centralizado')
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,2].set_xticks([])
    ax[0,2].set_yticks([])
    ax[0,3].set_xticks([])
    ax[0,3].set_yticks([])

    ax[1,0].imshow(np.log(1+np.abs(img40)),cmap='gray')
    ax[1,0].set_title('40 Original')
    ax[1,1].imshow(np.log(1+np.abs(img40_fourier1)),cmap='gray')
    ax[1,1].set_title('40 Amplitude')
    ax[1,2].imshow(np.log(1+np.abs(img40_fourier2)),cmap='gray')
    ax[1,2].set_title('40 Fases')
    ax[1,3].imshow(np.log(1+np.abs(img40_fourier3)),cmap='gray')
    ax[1,3].set_title('40 Centralizado')
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,2].set_xticks([])
    ax[1,2].set_yticks([])
    ax[1,3].set_xticks([])
    ax[1,3].set_yticks([])

    ax[2,0].imshow(np.log(1+np.abs(img_deslocada)),cmap='gray')
    ax[2,0].set_title('Deslocada Original')
    ax[2,1].imshow(np.log(1+np.abs(img_deslocada_fourier1)),cmap='gray')
    ax[2,1].set_title('Deslocada Amplitude')
    ax[2,2].imshow(np.log(1+np.abs(img_deslocada_fourier2)),cmap='gray')
    ax[2,2].set_title('Deslocada Fases')
    ax[2,3].imshow(np.log(1+np.abs(img_deslocada_fourier3)),cmap='gray')
    ax[2,3].set_title('Deslocada Centralizado')
    ax[2,0].set_xticks([])
    ax[2,0].set_yticks([])
    ax[2,1].set_xticks([])
    ax[2,1].set_yticks([])
    ax[2,2].set_xticks([])
    ax[2,2].set_yticks([])
    ax[2,3].set_xticks([])
    ax[2,3].set_yticks([])

    plt.show()

if __name__ == "__main__":
        main()