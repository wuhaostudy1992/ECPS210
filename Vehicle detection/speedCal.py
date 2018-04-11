# coding=utf-8
import numpy as np
import cv2

cap = cv2.VideoCapture("high.flv")

# Shi-Tomasi
feature_params = dict(maxCorners=10, qualityLevel=0.1, minDistance=1, blockSize=9)

# LK
lk_params = dict(winSize=(30, 30), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# create random color
color = np.random.randint(0, 255, (100, 3))

# get the first frame and turn to gery
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# ST
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

while 1:
    ret, frame = cap.read()

    if frame is None:
        cv2.waitKey(0)
        break
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算光流
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # 选择好的特征点
        if p1 is None:
            pass
        elif p0 is None:
            pass
        else:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # 输出每一帧内特征点的坐标
        # 坐标个数为之前指定的个数
        #print(good_new)

        # 绘制轨迹
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # 更新上一帧以及特征点
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
