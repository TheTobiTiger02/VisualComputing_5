import cv2
import numpy as np
from matplotlib import pyplot as plt


def plotHistogram(image, cumulative=False, title='Histogram'):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    if cumulative:
        hist = hist.cumsum()
    hist /= hist.sum()

    plt.figure()
    plt.plot(hist)
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.show()
    return hist


def equalize_histogram(image):
    # Step 1: Berechnung des Histogramms
    hist, bins = np.histogram(image.flatten(), 256, [0.0, 256.0])

    # Step 2: Berechnung des akkumulierten Histogramms
    cdf = hist.cumsum()

    # Schritt 3: CDF auf die höchste Pixelintensität skalieren
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = np.ma.filled(cdf, 0).astype('uint8')

    # Schritt 4: Wenden Sie die resultierende Abbildung auf jedes Pixel des Input-Bildes an
    img_equalized = cdf[image]

    return img_equalized


# Load the image
img = cv2.imread('schrott.png', cv2.IMREAD_GRAYSCALE)

# Original Histogram
plotHistogram(img, cumulative=False, title='Original Histogram')

# Apply Histogram Equalization
img_equalized = equalize_histogram(img)

# Equalized Histogram
plotHistogram(img_equalized, cumulative=False, title='Equalized Histogram')

# Compare Images
cv2.imshow('Original Image', img)
cv2.imshow('Equalized Image', img_equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
