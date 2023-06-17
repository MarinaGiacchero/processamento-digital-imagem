import cv2 
import numpy as np

# Carregar a imagem em escala de cinza
img = cv2.imread('img/flores.jpg', 0)

# 1. Segmentação de objetos:
ret, img_segmented = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 2. Detecção de bordas:
edges = cv2.Canny(img, 100, 200)

# 3. Extração de características:
ret, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
features = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Pré-processamento de imagens:
ret, img_processed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Exibir as imagens antes e depois
cv2.imshow('Imagem Original', img)
cv2.imshow('Segmentação de Objetos', img_segmented)
cv2.imshow('Detecção de Bordas', edges)
cv2.imshow('Extração de Características', img_binary)
cv2.imshow('Pré-processamento de Imagens', img_processed)
cv2.waitKey(0)
cv2.destroyAllWindows()