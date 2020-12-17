import cv2
import numpy as np
from utils import stackImages

def empty(a):
    pass

cv2.namedWindow('trackBars')
cv2.resizeWindow('trackBars', 640, 240)
cv2.createTrackbar('Hue min', 'trackBars', 0, 179, empty)
cv2.createTrackbar('Hue max', 'trackBars', 179, 179, empty)
cv2.createTrackbar('Sat min', 'trackBars', 48, 255, empty)
cv2.createTrackbar('Sat max', 'trackBars', 255, 255, empty)
cv2.createTrackbar('Val min', 'trackBars', 147, 255, empty)
cv2.createTrackbar('Val max', 'trackBars', 255, 255, empty)

while True:
    img = cv2.imread('resources/lambo.png')
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos('Hue min', 'trackBars')
    h_max = cv2.getTrackbarPos('Hue max', 'trackBars')
    s_min = cv2.getTrackbarPos('Sat min', 'trackBars')
    s_max = cv2.getTrackbarPos('Sat max', 'trackBars')
    v_min = cv2.getTrackbarPos('Val min', 'trackBars')
    v_max = cv2.getTrackbarPos('Val max', 'trackBars')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
        
    imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
    cv2.imshow('stecked', imgStack)
    cv2.waitKey(1)