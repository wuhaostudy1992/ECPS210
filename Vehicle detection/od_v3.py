import argparse
import datetime
import time
import cv2

cap = cv2.VideoCapture('high.flv')
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    #Get frame
    grabbed, frame = cap.read()
    text = "Unoccupied"
 
    # the end of video
    if not grabbed:
        break
 
    # turn to gray and gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
    
    # background substraction
    fgmask = fgbg.apply(frame)
    ret, binary = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
    
    # find contours
    fgmask, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # record total number of vehicles
    totalCount = 0
 
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 25:
            continue
        totalCount += 1
        # compute the bounding box for the contour, draw it on the frame and update the text
        # computer contours and draw contours
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #text = "Occupied"
    # draw the text on the frame
    cv2.putText(frame, "Total number of vehicles: {}".format(totalCount), (1000, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 
    #show
    cv2.imshow("Real time road", frame)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
 
    #Press ESC key
    if key == 27:
        break
 
#cleat and destory
cap.release()
cv2.destroyAllWindows()
