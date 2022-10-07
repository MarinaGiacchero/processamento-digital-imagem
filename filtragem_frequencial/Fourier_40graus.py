import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Imagem original
    img = np.ones((512, 512)).astype(int)
    img[200:315,200:315] = 255

    # Rotacionar em 40°
    imagem = Image.fromarray(img)
    imgOriginal= imagem.rotate(40)

    # Amplitude
    imgAmplitude = np.fft.fft2(imgOriginal)

    # Fases
    imgFtt = np.fft.fftshift(np.fft.fft2(imgOriginal))
    imgFases= np.angle(imgFtt)

    # Centralizado
    imgCentralizado = np.fft.fftshift(imgAmplitude)

    fig, ax = plt.subplots(nrows=1,ncols=4)
    ax[0].imshow(np.log(1+np.abs(imgOriginal)),cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(np.log(1+np.abs(imgAmplitude)),cmap='gray')
    ax[1].set_title('Amplitude')
    ax[2].imshow(np.log(1+np.abs(imgFases)),cmap='gray')
    ax[2].set_title('Fases')
    ax[3].imshow(np.log(1+np.abs(imgCentralizado)),cmap='gray')
    ax[3].set_title('Centralizado')

    plt.show()

if __name__ == "__main__":
        main()