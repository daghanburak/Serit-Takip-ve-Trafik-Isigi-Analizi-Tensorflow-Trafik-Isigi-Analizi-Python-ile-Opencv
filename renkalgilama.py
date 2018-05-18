#Ortam ışığı ve şekilsel etmenler programı çok fazla etkilemektedir. 

import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)

while True:
    # Görüntüyü aldık
    _, frame = cap.read()
    frame = imutils.resize(frame,width=250)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV nin içindeki renk aralıkları
    lower_yellow = np.array([20, 0, 0])
    upper_yellow = np.array([40, 255, 255])
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([20, 255, 255])
    lower_green = np.array([45, 100, 50])
    upper_green = np.array([75, 255, 255])

    # Yukarıda belırledıgımız eşik değerlerini gray goruntunun içinde eşleştirdik.
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    # bitwise and operatörü ile de ana goruntude yukarıda buldugumuz mask'i aldık.
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res1 = cv2.bitwise_and(frame, frame, mask=mask1)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)
    #res=res or res1  res2
    res=cv2.add(res,res1,res2)
    img = cv2.medianBlur(res, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 10, 20,
                               param1=50, param2=30, minRadius=30, maxRadius=45)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x=circles[0][1]
        y=circles[0][0]
        r=circles[0][2]
        pxs=frame[x,y]
        #print("px0",pxs[0])
        #print("px1",pxs[1])
        for i in circles[0:2]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cimg, (i[0], i[1]), 1, (0, 0, 255), 1)
        if 0<=pxs[0] and pxs[0]<20 and pxs[1]<255 and 0<pxs[1] :
            print("Dur")

        elif 45<=pxs[0] and pxs[0]<=75 and pxs[1]>=100 and pxs[1]<255:
            print("Geccc")

        else:
            print("Algılanamadı")
        cv2.imshow('hsv', hsv)
        cv2.imshow('detected circles', cimg)
        cv2.imshow('res', res)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
