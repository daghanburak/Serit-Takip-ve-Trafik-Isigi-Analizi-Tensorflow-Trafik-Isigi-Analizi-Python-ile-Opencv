from trafiklamba import merkezbul #trafiklamba modülünden merkezbul fonksiyonu  dahil edildi
import cv2                        #cv2 modülü dahil edildi.
import numpy as np                #numpY modülü np olarak dahil edildi.


img = cv2.imread('sarı.png',1)              #sarı.png , yesil.png,kırmızı.jpg fotolarından yazdığımız birini img adlı değişkene  RGB matris olarak aldık.
img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #img'ye aldığımız RGB resmi img1 adlı degiskene GRAY olarak cevirdik.
boundaries=[([0,100,100],[20,255,255]),([20,0,0],[40,255,255]),([45,100,50],[75,255,255])]  #burada renklerin upper-lower değerlerini dizi olarak boundaries adlı degıskene aldık.
print(boundaries[0][0][0],"---",boundaries[0][1][0])    # kırmızının upper-lower değeri
print(boundaries[1][0][0],"---",boundaries[1][1][0])    # sarının upper-lower değeri
print(boundaries[2][0][1],"---",boundaries[2][1][1])    # yesilin upper-lower değeri
j=merkezbul(img1)   #merkezbul fonksiyonuna img1 fotografını parametre olarak gönderdik ve fotografta bulunan trafik lambasının merkez bilgisini almış olduk.  
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #BGR olan img verisini hsv'ye HSV olarak atadık.
j=j[0]  # merkez bilgilerinin ilk indisinde kayıtlı olan bilgiler j ye atadık.
if len(j)==1:    # j dizisinin uzunlugunu aldık.
    for i in j:  # For dönügüne girdik
        #sırası ile yesil,sarı ve kırmızı rengi kontrol ederek hangi lambanın yandıgını if-else yapısı ile baktık.
        if int(i[1])>=270 and int(i[1])<290:    # yesil ışıgın fotograftaki lamba konumuna göre karsılastırarak kosul olusturduk.
            pxy=img[i[0],i[1]]   # i den bulunan lamba merkezlerinin pixel değerlerini pxy'e rgb olarak aldık.
            lower=np.array(boundaries[2][0],dtype="uint8")  #lower degiskenıne int tipinde yesil'in lower degerlerını aldık
            upper=np.array(boundaries[2][1],dtype="uint8")   #upper degiskenıne int tipinde yesil'in upper degerlerını aldık
            mask = cv2.inRange(hsv, lower, upper)    # lower ve upper aralıgına göre hsv fotografta yesil rengi taradık ve mask degıskenıne atadık.
            res = cv2.bitwise_and(img,img, mask= mask)  # bitwise and operatörü ile de ana goruntude yukarıda buldugumuz mask'i aldık.
            cv2.imshow("yesil",res)     # ana görüntüde buldugumuz mask'ı ana fotografımızda gösterdik
            print("Geccccccccc")    # Eger fotografta yeşil renk var ise gecccc yazacak

        elif int(i[1])>=180 and int(i[1])<190:  # sarı ışıgın fotograftaki lamba konumuna göre karsılastırarak kosul olusturduk.
            pxs=img[i[0],i[1]]      #i den bulunan lamba merkezlerinin pixel değerlerini pxs'e rgb olarak aldık.
            lower=np.array(boundaries[1][0],dtype="uint8")  #lower degiskenıne int tipinde sarı'nin lower degerlerını aldık
            upper=np.array(boundaries[1][1],dtype="uint8")   #upper degiskenıne int tipinde sarı 'nin upper degerlerını aldık
            mask = cv2.inRange(hsv, lower, upper) # lower ve upper aralıgına göre hsv fotografta sarı rengi taradık ve mask degıskenıne atadık.
            res = cv2.bitwise_and(img,img, mask= mask) # bitwise and operatörü ile de ana goruntude yukarıda buldugumuz mask'i aldık.
            cv2.imshow("sarı",res)   # ana görüntüde buldugumuz mask'ı ana fotografımızda gösterdik
            print("Hazırlannnnn")   # Eger fotografta sarı renk var ise Hazırlannn yazacak
            
        elif int(i[1])>=80 and int(i[1])<95: #kırmızı ışıgın fotograftaki lamba konumuna göre karsılastırarak kosul olusturduk.
            pxk=img[i[0],i[1]] #i den bulunan lamba merkezlerinin pixel değerlerini pxk'e rgb olarak aldık.
            lower=np.array(boundaries[0][0],dtype="uint8")  #lower degiskenıne int tipinde kırmızının'nin lower degerlerını aldık
            upper=np.array(boundaries[0][1],dtype="uint8")  #upper degiskenıne int tipinde kırmızın'nin upper degerlerını aldık
            mask = cv2.inRange(hsv, lower, upper)  # lower ve upper aralıgına göre hsv fotografta kırmızı rengi taradık ve mask degıskenıne atadık.
            res = cv2.bitwise_and(img,img, mask= mask)  # bitwise and operatörü ile de ana goruntude yukarıda buldugumuz mask'i aldık.
            cv2.imshow("kırmızı",res)   # ana görüntüde buldugumuz mask'ı ana fotografımızda gösterdik
            print("Durrrrrrrr")         # Eger fotografta kırmızı renk var ise Hazırlannn yazacak
            

        else: #eger herhangi bir renk algılamaz ise fotografta buraya girecek
            print("Trafik Lambası Algılanamadı")
else: # eger fotografta 2 den fazla ısık yanıyorsa buraya gırecek
    print("Trafik Lambasında birden fazla ışık algılandı")
        

cv2.waitKey(0)  #herhangı bir tuşa basana kadar bekleyecek
cv2.destroyAllWindows() #tüm pencereler kapanacak


