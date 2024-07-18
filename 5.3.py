import cv2
import numpy as np

# Bild laden
img = cv2.imread('kante.png', cv2.IMREAD_GRAYSCALE)

# Gaußsche Weichzeichnermatrix F1
F1 = np.array([[0, 0, 0],
               [-1, 1, 0],
               [0, 0, 0]])

# Faltung mit F1 und delta=128
result1 = cv2.filter2D(img, -1, F1, delta=128) # border-type ist default Zero-Padding ODER Edge Padding(!)

# Faltungsmatrix F2
F2 = np.array([[0, 0, 0],
               [0, -1, 1],
               [0, 0, 0]])

# Faltung des Ergebnisses mit F2 und delta=128
result2 = cv2.filter2D(result1, -1, F2, delta=128)

# Faltungsmatrix F3
F3 = np.array([[0, 0, 0],
               [1, -2, 1],
               [0, 0, 0]])

# Faltung des ursprünglichen Bildes mit F3 und delta=128
result3 = cv2.filter2D(img, -1, F3, delta=128)

# Faltungsmatrix F4
F4 = np.array([[0, 0, 0],
               [0.333, 0.333, 0.333],
               [0, 0, 0]])

# Faltung einer Kopie des ursprünglichen Bildes mit F4 und delta=128
result4 = cv2.filter2D(img.copy(), -1, F4, delta=128)

# Faltungsmatrix F5
F5 = np.array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])

# Faltung einer weiteren Kopie des ursprünglichen Bildes mit F5 und delta=128
result5 = cv2.filter2D(img.copy(), -1, F5, delta=128)

# Faltungsmatrix F6
F6 = F4 - F5

# Faltung des ursprünglichen Bildes mit F6 und delta=128
result6 = cv2.filter2D(img, -1, F6, delta=128)

# Bilder anzeigen
cv2.imshow('Original', img)
cv2.imshow('Result 1', result1)
cv2.imshow('Result 2', result2)
cv2.imshow('Result 3', result3)
cv2.imshow('Result 4', result4)
cv2.imshow('Result 5', result5)
cv2.imshow('Result 6', result6)

# Bilder speichern
cv2.imwrite('result1.png', result1)
cv2.imwrite('result2.png', result2)
cv2.imwrite('result3.png', result3)
cv2.imwrite('result4.png', result4)
cv2.imwrite('result5.png', result5)
cv2.imwrite('result6.png', result6)

# Warten auf eine Taste, dann schließen
cv2.waitKey(0)
cv2.destroyAllWindows()