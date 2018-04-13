import cv2
import numpy as np

def merkezbul(img):     #ana modülden gelen gray fotoyu img'ye aldık
    img = cv2.medianBlur(img,5)     # görüntüye median filtresi uyguladık
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)     # görüntüyü GRAY'dan BGR'a cevirdik

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=35,maxRadius=40)
    # Houghcircles komutu ile minimum maxımum yarıçaplara göre daireleri bulduk ve biilgilerini circles degiskenıne aldık
    circles = np.uint16(np.around(circles)) # circles icindeki degerleri int yaptık
    for i in circles[0,:]: # icine gelen daire bilgilerini for dönügüsi içinde çizdirdik
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2) # cemberin dışını çizdirdik.
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3) # cemberın merkezıni cizdirdik
    cv2.imshow("Trafik Lambası Cember",cimg)  # cimg fotografı uzerınde cemberlerı gösterdik
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return circles  #circles da bulunan dairenin bilgilerini ana fonksiyona dönderdik.



