import argparse
import datetime
import time
import cv2
import numpy as np
import sys

'''
Adding an array to store all boxes of all vehicles
'''

ReshapeSize = 2
frameCenter = (960//ReshapeSize, 440//ReshapeSize)
WIDTH = 1920//ReshapeSize
HEIGHT = 1080//ReshapeSize

USERECT = True
SHOWRECT = True
SHOWRECTCENTER = False

USECIRCLE = not USERECT
SHOWCIRCLE = False
SHOWCIRCLECENTER = False

SHOWDIVIDELINE = True



def CenterCalculation(frame, contour, bindingboxes):
    center = [0, 0]
    if USERECT:
        # compute the bounding box for the contour, draw it on the frame and update the text
        # computer contours and draw contours
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        center = tuple((box[1] + box[3])//2)
        if SHOWRECT:
            cv2.drawContours(frame,[box],0,(0,0,255),2)
        if SHOWRECTCENTER:
            cv2.circle(frame, center, 3, (0, 255, 255), -1)
        bindingboxes.append(box)
            
    if USECIRCLE:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
        if SHOWCIRCLE:
            cv2.circle(frame,center,radius,(0,255,0),2)
        if SHOWCIRCLECENTER:
            cv2.circle(frame, center, 3, (0, 255, 255), -1)
    return center

def NumberCalculation(count, center):
    count[0] += 1
    if center[0] < frameCenter[0]:
        if center[1] < frameCenter[1]:
            count[1] += 1
        else:
            count[2] += 1
    else:
        if center[1] < frameCenter[1]:
            count[3] += 1
        else:
            count[4] += 1

def DrawText(frame, count):
    '''
    Draw text on the screen.
    count: the number of vehicles of current frame, [totalCount, topleft, topright, bottomleft, bottomright]
    totalCount
    
    cv2.putText(frame, "Total number of vehicles: {}".format(count[0]), (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of topleft: {}".format(count[1]), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of topright: {}".format(count[2]), (900, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of bottomleft: {}".format(count[3]), (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total number of vehicles of bottomright: {}".format(count[4]), (900, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    '''
    
    cv2.putText(frame, "{}".format(count[0]), (WIDTH//2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(count[1]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(count[2]), (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(count[3]), (WIDTH-10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(count[4]), (WIDTH - 10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)




if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print("Please specific the path of the video.")
        sys.exit()
    
    #Initial the tracking
    tracker = cv2.TrackerMIL_create()
    #Store all binding boxes and trackers
    bindingboxes = []
    trackers = []
    
    cap = cv2.VideoCapture(sys.argv[1])
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    
    ret, frame = cap.read()
    
    #use the first frame to compare, but may not be available
    firstFrame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    
    while True:
        #Get frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        
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
        if SHOWDIVIDELINE:
            cv2.line(frame,(0,frameCenter[1]),(WIDTH,frameCenter[1]),(255,0,0),1)
            cv2.line(frame,(frameCenter[0], 0),(frameCenter[0], HEIGHT),(255,0,0),1)
        
        # find contours
        fgmask, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # record total number of vehicles
        count = [0, 0, 0, 0, 0] #totalCount, topleft, topright, bottomleft, bottomright
        # all vehicles' last location []
        lastLocation = []
     
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 20:
                continue
            
            # center of contour
            center = CenterCalculation(frame, c, bindingboxes)

            NumberCalculation(count, center)
            '''todo: calculate the speed of all vehicles'''
        
        # Update tracker
        ok, bbox = tracker.update(frame)
        
        # draw the text on the frame
        DrawText(frame, count)

        #show
        cv2.imshow("Real time road", frame)

        key = cv2.waitKey(1) & 0xFF 
        #Press ESC key
        if key == 27:
            break
     
    #clear and destory
    cap.release()
    cv2.destroyAllWindows()
