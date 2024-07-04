import numpy as np ## optional
import cv2
print("OpenCV-Version: " + cv2.__version__)
img = cv2.imread("yoshi.png")
img2 = cv2.imread("yoshi.png")
# print image width and height
print("Image width: " + str(img.shape[1]))
print("Image height: " + str(img.shape[0]))

# print image color channels
print("Image channels: " + str(img.shape[2]))

# Bilddatenformat auf float setzen
img2 = img2.astype(np.float32)


# Draw a red 10x10 rectangle in the middle
cv2.rectangle(img, (img.shape[1]//2-5, img.shape[0]//2-5), (img.shape[1]//2+5, img.shape[0]//2+5), (0, 0, 255), -1)

# Replace every 5th row with a black pixel
for i in range(0, img.shape[0], 5):
    img[i] = [0, 0, 0]
    
# Save the image
cv2.imwrite("yoshi_modified.png", img)

# Display the image
cv2.imshow("uint8", img)
cv2.imshow("float32", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()