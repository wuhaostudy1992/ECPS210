import argparse
import datetime
import time
import cv2
import numpy as np

'''
New: Divide into four areas
'''

frameCenter = (635, 356)
WIDTH = 720
HEIGHT = 1280

SHOWRECT = True

cap = cv2.VideoCapture('high.flv')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    #Get frame
    ret, frame = cap.read()
 
    # the end of video
    if not ret:
        break
 
    # turn to gray and gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    # background substraction
    fgmask = fgbg.apply(frame)
    ret, binary = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
    
    # draw the divide lines
    cv2.line(frame,(0,frameCenter[1]),(1280,frameCenter[1]),(255,0,0),1)
    cv2.line(frame,(frameCenter[0], 0),(frameCenter[0], 720),(255,0,0),1)
    
    # find contours
    fgmask, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # record total number of vehicles
    totalCount = 0
    topleft, topright, bottomleft, bottomright = 0, 0, 0, 0
    # all vehicles' last location []
    lastLocation = []
 
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 50:
            continue
        totalCount += 1
        # center of contour
        center = [0, 0]
        if SHOWRECT:
            # compute the bounding box for the contour, draw it on the frame and update the text
            # computer contours and draw contours
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,0,255),2)
            center = tuple((box[1] + box[3])//2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
        else:
            (x,y),radius = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(frame,center,radius,(0,255,0),2)
        
        if center[0] < frameCenter[0]:
            if center[1] < frameCenter[1]:
                topleft += 1
            else:
                topright += 1
        else:
            if center[1] < frameCenter[1]:
                bottomleft += 1
            else:
                bottomright += 1
        '''todo: calculate the speed of all vehicles'''
        
        
    # draw the text on the frame
    cv2.putText(frame, "Total number of vehicles: {}".format(totalCount), (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of topleft: {}".format(topleft), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of topright: {}".format(topright), (900, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of bottomleft: {}".format(bottomleft), (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of bottomright: {}".format(bottomright), (900, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
    
    #show
    cv2.imshow("Real time road", frame)

    key = cv2.waitKey(1) & 0xFF 
    #Press ESC key
    if key == 27:
        break
 
#clear and destory
cap.release()
cv2.destroyAllWindows()
