import numpy as np
import cv2

'''BackgroundSubtractor'''
cap = cv2.VideoCapture('test.flv')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
