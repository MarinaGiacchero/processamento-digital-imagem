#Exemplo de leitura e plot de imagens
#Bibliotecas Numpy, Pillow, Matplotlib

from turtle import width
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img= Image.open('img/lena_gray_512.tif')
    #img= Image.open('img/house.tif')
    #img= Image.open('img/cameraman.tif')
    print(img.size)
    
    #4 quadrados brancos
    imgFour= np.array(img)
    imgFour[0:10,0:10] = 255
    imgFour[501:511,501:511] = 255
    imgFour[0:10,501:511] = 255
    imgFour[501:511,0:10] = 255

    #quadrado preto no meio
    imgDimen= np.array(img)
    imgDimen[250:265,250:265] = 0

    #negativo da imagem
    npImg= np.array(img)
    npImg= 255-npImg  
    imgNew= Image.fromarray(npImg)
    #metade da intensidade
    imgInt=np.array(img)
    imgInt= imgInt/2
    imgInt= Image.fromarray(imgInt) 

    fig, ax = plt.subplots(nrows=2,ncols=3)
    ax[0,0].imshow(img,cmap='gray')
    ax[0,0].set_title('Imagem original')
    ax[0,1].imshow(imgNew, cmap='gray')
    ax[0,1].set_title("Imagem negativa")
    ax[1,0].imshow(imgInt, cmap='gray')
    ax[1,0].set_title("Imagem com metade da intensidade")
    ax[1,2].imshow(imgFour, cmap='gray')
    ax[1,2].set_title("Imagem com 4 quadrados")
    ax[0,2].imshow(imgDimen, cmap='gray')
    ax[0,2].set_title("Imagem com 1 quadrado preto")
    plt.show()   
    
    
if __name__ == "__main__":
    main()