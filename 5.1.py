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


def linear_contrast_transform(image, min_val, max_val):
    # Clip the values and scale to [100, 150]
    transformed = np.clip(image, min_val, max_val)
    transformed = ((transformed - min_val) / (max_val - min_val)) * 50 + 100
    return transformed.astype(np.uint8)


def maximize_contrast(image):
    # Scale to [0, 255]
    min_val = np.min(image)
    max_val = np.max(image)
    transformed = ((image - min_val) / (max_val - min_val)) * 255   # Folie Seite 40
    return transformed.astype(np.uint8)


# Load the image
img = cv2.imread('schrott.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Original Histogram
plotHistogram(img, cumulative=False, title='Original Histogram')

# Step 2: Cumulative Histogram
plotHistogram(img, cumulative=True, title='Cumulative Histogram')

# Step 3: Linear Grauwert-Transformation (Contrast Reduction)
min_gray = np.min(img)
max_gray = np.max(img)
img_low_contrast = linear_contrast_transform(img, min_gray, max_gray)

# Step 4: Histogram of Low Contrast Image
plotHistogram(img_low_contrast, cumulative=False, title='Low Contrast Histogram')

# Step 5: Maximize Contrast of Low Contrast Image
img_max_contrast = maximize_contrast(img_low_contrast)

# Step 6: Histogram of Max Contrast Image
plotHistogram(img_max_contrast, cumulative=False, title='Max Contrast Histogram')

# Compare Images
cv2.imshow('Original Image', img)
cv2.imshow('Low Contrast Image', img_low_contrast)
cv2.imshow('Max Contrast Image', img_max_contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
