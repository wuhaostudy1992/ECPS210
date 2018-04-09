import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

'''BackgroundSubtractor'''
cap = cv2.VideoCapture('test.flv')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    ret, thresh = cv2.threshold(fgmask,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ret, binary = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
      
    fgmask, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    backtorgb = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB)
      
    cv2.drawContours(backtorgb,contours,-1,(0,0,255),3)  
      
   
    cv2.imshow('frame',backtorgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


'''img = cv2.imread('coins.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)'''
