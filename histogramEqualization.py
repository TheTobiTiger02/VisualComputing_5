import cv2
from matplotlib import pyplot as plt
import numpy as np


def histEqualization(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    histcum = hist.cumsum()
    hist /= hist.sum()

    cdf = histcum / histcum.max()

    cdf = np.floor(cdf * 255).astype(np.uint8)

    equalized_image = cdf[image]

    return equalized_image

def plotHistogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()

    plt.figure()
    plt.plot(hist)
    plt.show()
    return hist


img = cv2.imread('schrott.png')
min_val = np.min(img)
max_val = np.max(img)
scaled_img = np.clip(100 + (img - min_val) * (50 / (max_val - min_val)), 100, 150).astype(np.uint8)
eqImg = histEqualization(img)
eqScaledImg = histEqualization(scaled_img)

plt.figure()
plt.title('Equalized Image')
plt.imshow(eqImg, cmap='gray')
plt.axis('off')
plt.show()

plt.figure()
plt.title('Scaled Image')
plt.imshow(eqScaledImg, cmap='gray')
plt.axis('off')
plt.show()

plotHistogram(eqImg)
plotHistogram(eqScaledImg)