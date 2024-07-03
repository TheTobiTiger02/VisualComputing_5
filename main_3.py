import cv2
import numpy as np

def on_trackbar(val):
    global hsv_img, modified_img
    hsv_img[white_pixels] = [val, 255, 255]
    modified_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    cv2.imshow("Modified Yoshi", modified_img)

# Load images
img = cv2.imread("yoshi.png")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# Convert Yoshi image to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Find white pixels in the mask
white_pixels = np.where(mask == 255)

# Create a window for the image
cv2.namedWindow("Modified Yoshi")

# Create a trackbar for H-value adjustment
cv2.createTrackbar("H-Value", "Modified Yoshi", 120, 255, on_trackbar)

# Initial display
modified_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
cv2.imshow("Modified Yoshi", modified_img)

cv2.waitKey(0)
cv2.destroyAllWindows()