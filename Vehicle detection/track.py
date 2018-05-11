import sys
import copy
import argparse
import cv2
import numpy as np

from entity import Entity

X = 1280 #852
Y = 720 #480
topleft = (400, 160)
topright = (550, 160)
bottomleft = (400, 240)
bottomright = (500, 220)


def overlap(box1, box2):
    """
    Check the overlap of two boxes
    """
    endx = max(box1[0] + box1[2], box2[0] + box2[2])
    startx = min(box1[0], box2[0])
    width = box1[2] + box2[2] - (endx - startx)

    endy = max(box1[1] + box1[3], box2[1] + box2[3])
    starty = min(box1[1], box2[1])
    height = box1[3] + box2[3] - (endy - starty)

    if (width <= 0 or height <= 0):
        return 0
    else:
        Area = width * height
        Area1 = box1[2] * box1[3]
        Area2 = box2[2] * box2[3]
        ratio = Area / (Area1 + Area2 - Area)
        #print(ratio)
        return ratio
        
def ShowLine(frame, show):
    '''
    Show the boundary
    '''
    cv2.line(frame,(0, topleft[1]),topleft,(255,0,0),1)
    cv2.line(frame,topleft,(topleft[0], 0),(255,0,0),1)
    #cv2.line(frame, topleft, topleft, (160,160,160),3)
    
    cv2.line(frame,(topright[0], 0),topright,(255,0,0),1)
    cv2.line(frame,topright,(X, topright[1]),(255,0,0),1)
    
    cv2.line(frame,(0, bottomleft[1]),bottomleft,(255,0,0),1)
    cv2.line(frame,bottomleft,(bottomleft[0], Y),(255,0,0),1)
    
    cv2.line(frame,(bottomright[0], Y),bottomright,(255,0,0),1)
    cv2.line(frame,bottomright,(X, bottomright[1]),(255,0,0),1)

def UpdateCount(frame, totalcount, x, y, remove=False):
    '''
    Update the count of the current frame and show
    '''
    if remove:
        totalcount[0] -= 1
        if 0 < x < topleft[0] and 0 < y < topleft[1]:
            totalcount[1] -= 1
        elif topright[0] < x < X and 0 < y < topright[1]:
            totalcount[2] -= 1
        elif 0 < x < bottomleft[0] and bottomleft[1] < y < Y:
            totalcount[3] -= 1
        elif bottomright[0] < x < X and bottomright[1] < y < Y:
            totalcount[4] -= 1
    else:
        totalcount[0] += 1
        if 0 < x < topleft[0] and 0 < y < topleft[1]:
            totalcount[1] += 1
        elif topright[0] < x < X and 0 < y < topright[1]:
            totalcount[2] += 1
        elif 0 < x < bottomleft[0] and bottomleft[1] < y < Y:
            totalcount[3] += 1
        elif bottomright[0] < x < X and bottomright[1] < y < Y:
            totalcount[4] += 1

def DrawText(frame, totalcount):
    #cv2.putText(frame, "{}".format(totalcount[0]), (X//2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(totalcount[1]), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(totalcount[2]), (X-10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(totalcount[3]), (10, Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "{}".format(totalcount[4]), (X-10, Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def track(video, iou):
    camera = cv2.VideoCapture(video)
    res, frame = camera.read()
    print(frame.shape)
    y_size = frame.shape[0]
    x_size = frame.shape[1]

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    frames = 0
    counter = 0

    track_list = []
    #cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    
    while True:
        res, frame = camera.read()
        if not res:
            break
        #ShowLine(frame, True)    
        
        # Train the MOG2 with first frames frame
        fg_mask = fgbg.apply(frame)

        # Expansion and denoising the original frame
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check the bouding boxs
        totalcount = [0, 0, 0, 0, 0] #totalCount, topleft, topright, bottomleft, bottomright
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            
            if 0 < cx < topleft[0] and 0 < cy < topleft[1]:
                if cv2.contourArea(c) < 400:
                    continue
                totalcount[1] += 1
            elif topright[0] < cx < X and 0 < cy < topright[1]:
                if cv2.contourArea(c) < 200:
                    continue
                totalcount[2] += 1
            elif 0 < cx < bottomleft[0] and bottomleft[1] < cy < Y:
                if cv2.contourArea(c) < 1000:
                    continue
                totalcount[3] += 1
            elif bottomright[0] < cx < X and bottomright[1] < cy < Y:
                if cv2.contourArea(c) < 400:
                    continue
                totalcount[4] += 1
            else:
                continue
            
            '''if cv2.contourArea(c) < 150:
                continue'''
            
            # Extract roi
            img = frame[y: y + h, x: x + w, :]
            rimg = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            image_data = np.array(rimg, dtype='float32')
            image_data /= 255.
            roi = np.expand_dims(image_data, axis=0)
            
            e = Entity(counter, (x, y, w, h), frame)
            
            # Exclude existing targets in the tracking list
            if track_list:
                count = 0
                num = len(track_list)
                for p in track_list:
                    if overlap((x, y, w, h), p.windows) < iou:
                        count += 1
                    else:
                        
                        # TODO calculate speed and direction
                        #centerX = p.windows[0] + p.windows[1]//2
                        #centerY = p.windows[2] + p.windows[3]//2
                        #speed = (cx -centerX)^2 + (cy-centerY)^2
                        speed = np.sqrt(pow(x - p.windows[0], 2) + pow(y - p.windows[1], 2))//1
                        p.updateSpeed(speed)
                if count == num:
                    track_list.append(e)
            else:
                track_list.append(e)
            counter += 1
            #UpdateCount(frame, totalcount, x, y)
        
        DrawText(frame, totalcount)
        # Check and update goals
        if track_list:
            #tlist = copy.copy(track_list)
            for e in track_list:
                x, y = e.center
                if 10 < x < x_size - 10 and 10 < y < y_size - 10:
                    e.update(frame)
                    if e.visited == False:
                        e.speed = 0
                    else:
                        e.visited = False
                else:
                    track_list.remove(e)
            
        frames += 1
        
        #show
        cv2.imshow("Real time road", frame)

        key = cv2.waitKey(1) & 0xFF 
        #Press ESC key
        if key == 27:
            break
    
    #clear and destory
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    track(sys.argv[1], 0.3)
