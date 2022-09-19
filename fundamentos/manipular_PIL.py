#pillow

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
 
    img = Image.open('img/lena_gray_512.tif') #para utilizar a imagem da mulher
    #img= Image.open('img/house.tif') #para utilizar a imagem da casa
    #img= Image.open('img/cameraman.tif') #para utilizar a imagem do cameraman
    imgOriginal= img

    # para reduzir em 1.5x
    width  = int( img.size[0]*50/100 ) 
    height = int( img.size[1]*50/100 )
    new_size = (width, height)
    imgReduzida = img.resize(new_size, Image.Resampling.LANCZOS)

    # para aumentar em 2.5x
    width  = int( img.size[0]*250/100 ) 
    height = int( img.size[1]*250/100 )
    new_size = (width, height)
    imgAumentada = img.resize(new_size, Image.Resampling.LANCZOS)

    # para rotacionar 45°
    imagem45= img.rotate(45)    
    # para rotacionar 90°
    imagem90= img.rotate(90)
    # para rotacionar 100°
    imagem100= img.rotate(100)
    
    # para fazer a translação
    imgDeslocar=img.rotate(0, translate=(100,20))
    imgDeslocarE=img.rotate(0, translate=(35,45))

    fig, ax = plt.subplots(nrows=3,ncols=3)
    ax[0,0].imshow(imgReduzida,cmap='gray')
    ax[0,0].set_title('Imagem Reduzida')
    ax[0,1].imshow(imgAumentada,cmap='gray')
    ax[0,1].set_title('Imagem Aumentada')
    ax[1,1].imshow(imagem45,cmap='gray')
    ax[1,1].set_title('Imagem rotacionada a 45 graus')
    ax[1,0].imshow(imagem90,cmap='gray')
    ax[1,0].set_title('Imagem rotacionada a 90 graus')
    ax[0,2].imshow(imagem100,cmap='gray')
    ax[0,2].set_title('Imagem rotacionada a 100 graus')
    ax[1,2].imshow(imgDeslocar,cmap='gray')
    ax[1,2].set_title('Imagem com translação')
    ax[2,0].imshow(imgDeslocarE,cmap='gray')
    ax[2,0].set_title('Imagem com translação 35x45')
    ax[2,2].imshow(imgOriginal,cmap='gray')
    ax[2,2].set_title('Imagem original')
    plt.show()  
if __name__ == "__main__":
    main()