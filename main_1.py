# import numpy as np ## optional
import cv2
print("OpenCV-Version: " + cv2.__version__)
img = cv2.imread("yoshi.png")
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()