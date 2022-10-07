import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Imagem original
    img = np.ones((512, 512)).astype(int)
    img[200:315,200:315] = 255

    # Translação nos eixos x e y 
    imagem = Image.fromarray(img)
    imgAlterada = imagem.rotate(0, translate=(50,200))

    # Amplitude
    imgAlteradaAmplitude = np.fft.fft2(imgAlterada) 

    # Fases
    imgAlteradaFtt = np.fft.fftshift(np.fft.fft2(imgAlterada))
    imgAlteradaFases= np.angle(imgAlteradaFtt)

    # Centralizado
    imgAlteradaCentralizado = np.fft.fftshift(imgAlteradaAmplitude)


    fig, ax = plt.subplots(nrows=1,ncols=4)
    ax[0].imshow(np.log(1+np.abs(imgAlterada)),cmap='gray')
    ax[0].set_title('Deslocada Original')
    ax[1].imshow(np.log(1+np.abs(imgAlteradaAmplitude)),cmap='gray')
    ax[1].set_title('Deslocada Amplitude')
    ax[2].imshow(np.log(1+np.abs(imgAlteradaFases)),cmap='gray')
    ax[2].set_title('Deslocada Fases')
    ax[3].imshow(np.log(1+np.abs(imgAlteradaCentralizado)),cmap='gray')
    ax[3].set_title('Deslocada Centralizado')

    plt.show()

if __name__ == "__main__":
        main()