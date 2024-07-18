import cv2
from matplotlib import pyplot as plt
import numpy as np


def plotHistogram(image, cumulative=False):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    if cumulative:
        hist = hist.cumsum()
    hist /= hist.sum()

    plt.figure()
    plt.plot(hist)
    plt.show()
    return hist

img = cv2.imread('schrott.png')


# Scale the image to the range 100 to 150
min_val = np.min(img)
max_val = np.max(img)
scaled_img = np.clip(100 + (img - min_val) * (50 / (max_val - min_val)), 100, 150).astype(np.uint8)

#Scale the image to 0 255
max_contrast_img = np.clip((scaled_img - 100) * (255 / 50), 0, 255).astype(np.uint8)

plt.figure()
plt.title('normal')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

plt.figure()
plt.title('scaled 100 150')
plt.imshow(scaled_img, cmap='gray')
plt.axis('off')
plt.show()

plt.figure()
plt.title('max Contrast')
plt.imshow(max_contrast_img, cmap='gray')
plt.axis('off')
plt.show()


hist = plotHistogram(img, True)
hist2 = plotHistogram(max_contrast_img, True)
