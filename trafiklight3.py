import cv2
import numpy as np

def merkezbul():
    img = cv2.imread('trafik.jpg',0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=35,maxRadius=40)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #cv2.imshow("sss",cimg)
    #j=circles
    #kırmızı=j[0][1]
    #sarı=j[0][2]
    #yesil=j[0][0]
    #print(kırmızı)
    #print(sarı)
    #print(yesil)
    #cv2.imshow('detected circles',img)
    #cv2.imshow('detected circles',cimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return circles 

#merkezbul()
#yesil=j[0][0]
#kırmızı=j[0][1]
#sarı=j[0][2]
#cv2.waitKey(0)
#cv2.destroyAllWindows()

