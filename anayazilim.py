from trafiklight3 import merkezbul 
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('trafik.jpg',1)
cv2.imshow("Orjinal Trafik Lambası",img)
j=merkezbul()
yesil=j[0][0]
kırmızı=j[0][1]
sarı=j[0][2]
pxk=img[kırmızı[0],kırmızı[1]]#[10 12 12] kırmızının piksel değerleri bunun içerisinde
pxy=img[yesil[0],yesil[1]]#[134 136 136]
pxs=img[sarı[0],sarı[1]]#[33 35 35]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
boundaries=[([0,100,100],[20,255,255]),([20,0,0],[40,255,255]),([45,100,50],[75,255,255])]
for (lower,upper) in boundaries:
        if int(pxk[0])>int(lower[0]) and int(pxk[0])<=int(upper[0]):
            lower=np.array(lower,dtype="uint8")
            upper=np.array(upper,dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            # bitwise and operatörü ile de ana goruntude yukarıda buldugumuz mask'i aldık.
            res = cv2.bitwise_and(img,img, mask= mask)
            #ayarladıgımız 3 görüntüyü gösterdik
            cv2.imshow("kırmızı",res)
            print("Durrrrrrrr")
        if int(pxy[1])>int(lower[1]) and int(pxy[1])<=int(upper[1]):
            lower=np.array(lower,dtype="uint8")
            upper=np.array(upper,dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            # bitwise and operatörü ile de ana goruntude yukarıda buldugumuz mask'i aldık.
            res2 = cv2.bitwise_and(img,img, mask= mask)
            #ayarladıgımız 3 görüntüyü gösterdik
            cv2.imshow("yesil",res2)
            print("Geccccccccc")
        if int(pxs[0])>int(lower[0]) and int(pxs[0])<=int(upper[0]):
            lower=np.array(lower,dtype="uint8")
            upper=np.array(upper,dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            # bitwise and operatörü ile de ana goruntude yukarıda buldugumuz mask'i aldık.
            res3 = cv2.bitwise_and(img,img, mask= mask)
            #ayarladıgımız 3 görüntüyü gösterdik
            cv2.imshow("sarı",res3)
            print("Hazırlannnnn")
cv2.waitKey(0)
cv2.destroyAllWindows()
