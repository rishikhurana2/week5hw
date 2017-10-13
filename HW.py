import cv2
import math
import numpy as np

file = "hwimg.png"
fileLocation = "hwimg.png"
img = cv2.imread(fileLocation)
Cimg = cv2.imread(file)

cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("HSV Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Threshed Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Contour Image", cv2.WINDOW_AUTOSIZE)

cv2.imshow("Original Image", img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", img_hsv)

THRESHOLD_MIN = np.array([5, 50, 50], np.uint8)
THRESHOLD_MAX = np.array([15, 255, 255], np.uint8)
frame_threshold = cv2.inRange(img_hsv, THRESHOLD_MIN, THRESHOLD_MAX)

frame_threshed = cv2.inRange(img_hsv, THRESHOLD_MIN, THRESHOLD_MAX)
images, contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = -1
cv2.drawContours(Cimg, contours, count, (255, 255, 255), 10)

for cont in contours:
	approx = cv2.approxPolyDP(cont, 0.1*cv2.arcLength(cont, True), True)
	if (len(approx) == 4 and cv2.contourArea(cont) > 10000):
		cv2.imshow("Contour Image", Cimg)
cv2.imshow("Threshed Image", frame_threshold)
cv2.waitKey(0)
