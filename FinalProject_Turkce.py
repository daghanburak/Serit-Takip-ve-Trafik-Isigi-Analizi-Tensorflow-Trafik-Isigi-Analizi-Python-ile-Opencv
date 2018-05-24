import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
"""Eğitim dizini, 00000 ile 00061 arasındaki sıralı sayısal isimleri olan alt dizinler içerir.
Dizinin adı, 0 ile 61 arasındaki etiketleri temsil eder ve her dizindeki görüntüler,
o etikete ait trafik işaretlerini temsil eder. Görüntüler, çok yaygın olmayan .ppm formatında kaydedilir,
ancak neyse ki, bu format, skimage kütüphanesinde desteklenir.
"""
#Eğitilmiş DATASET: http://btsd.ethz.ch/shareddata/
def data_yükle (data_dir):
    """Bir veri seti yüklenir ve iki liste olarak geri dönderir:
    images: Her biri bir görüntüyü temsil eden Numpy dizileri dönderir.
    labels: Resimlerin etiketlerini temsil eden numaraların listesi.
    """
    # Tüm data_dir altındaki dizinleri aldık.Her bir data bir etiketi temsil ediyor..
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]

    #Etiket dizinlerinde dolaşıp ve dataları iki liste olarak, etiket ve görsel olarak topladık.
    labels = []
    images = []

    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        #Her etiket için, görüntüleri yükleyin ve bunları resim listesine ekleyin.
        #Ve etiket listesine etiket numarasını (yani dizin adı) ekleyin.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Eğitim ve tes verilerini yüklemek için veri dosyalarını eklemek için yollarını atadık.

dosya_yolu= "/traffic"
egitim_veri = os.path.join(dosya_yolu, "datasets/BelgiumTS/Training")
test_veri = os.path.join(dosya_yolu, "datasets/BelgiumTS/Testing")

images, labels = data_yükle(egitim_veri) #Burada eğitim verimizde bulunan tüm ggörüntü ve etiketlerini sıralı bir şekilde aldık.
print("Benzersiz Etiketler: {0}\n Toplam Görüntüler: {1}".format(len(set(labels)), len(images)))

def images_ve_labels_goster(images, labels):
    """Her bir etiketin ilk resmini görüntülüyoruz"""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Her etiket için ilk resmi seçiyoruz.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # 8 satı 8 sütün olarak
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        #Her bir etiketten bir fotoğraf ve parantez içinde o etiketteki toplam fotograf sayısı
        i += 1
        _ = plt.imshow(image) # örnek bir fotograf goster.
    plt.show()


#images_ve_labels_goster(images, labels)
def goster_label_images(images, label):
    """Berlirli bir etiketin goruntulerını gosterdık"""
    limit = 24  # maksımum 24 göruntu olsun dedik
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

"""
Ancak, görüntüler kare kare olsa da, hepsi aynı boyutta değiller. Farklı boy oranları var.
Basit sinir ağımız sabit boyutlu bir girdi alır, bu yüzden biraz ön işlem yapmalıyız.
Yakında anlayacağız, ancak önce bir etiket seçip daha fazla görsel görelim. 32 numaralı etiketi seçelim:
"""
goster_label_images(images, 40)
# images bir daha boyutlandırmadan önce boyutlarımıza bakalım.
for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))


# İmagesleri birdaha boyutlandırdık.
images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]

for image in images32[:5]:# Kontrol edelim goruntulerı
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))


labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)
#Shape ozelligigenellikle bir dizinin geçerli şeklini almak için kullanılır, ancak diziyi bir dizi boyut ölçüsü atayarak yerinde yeniden şekillendirmek için de kullanılabilir.


                    # Modeli tutmak icin bir graph adında nesne olusturduk.
graph = tf.Graph() #Graflar Tensorflow’un temelidir denilebilir. Çünkü her bir hesaplama/işlem/değişken graf üzerinde yapılmaktadır.


with graph.as_default():
    #images ve labels için yer tutucular yani placeholders tanımladık.Bu TensorFlow'un girdi alma şeklidir.Birden çok girdi verdiimizde bunu kullanmamız gerekıyor.

    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3]) #[None,yüksekli,genişlik,kanallar]
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten  a yer tutuculardan giriş yapıyıyoruz: [None, height, width, channels]
    # ve bunu şu şekle değiştirmesini sağlıyor bıze: [None, height * width * channels] == [None, 3072]
    #[None,3072] şeklinde düzleştirilmiş bir sonuç veriyor.

    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer. fully_connected creates a variable called weights, representing a fully connected weight matrix, which is multiplied by the inputs to produce a Tensor of hidden units. 
    # Generates logits of size [None, 62]
    #operasyon sonucunu temsil eden bir değişken dondurur.
    logits1 = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    #fully_connected,bağlı birimlerin bir Tensor'u üretmek için girdilerle çarpılan,tamamen bir weight matrisi temsil eden,wieghts bir degişkeni oluşturur.

    # Logits değerlerini label index sine int değer dönderir.
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits1, 1) #1D uzunluktaki vektörün en büyük değerini bulur.

    # Define the loss function. 
    # Cross-entropy is a good choice for classification.Günlükleri ve etiketler arasında seyrek softmax çapraz entropisini hesaplar.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=labels_ph))#Bir tensörün boyutları boyunca elemanların ortalamasını hesaplar.


    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) #öğrenme oranımızı belirledik ve kayan nokta olarak bu degerı kulllandık.
    #minimize ile kaybı en aza indirmek için işlemler yaparız.

    # And, finally, an initialization op to execute before training.
    # TODO: rename to tf.global_variables_initializer() on TF 0.12.
    init = tf.initialize_all_variables() #Tensorflow içerisinde yukarıdaki değişkenlerin kullanılabilmesi için “initialize” edilmesi gerekmektedir. 
"""
print("[None,3072] images_flat: ", images_flat)
print("logits: ", logits1)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)
"""
# TRAINING
# Yukarda oluşturdugumuz grapph ı çalıştırmak için session başlatıyoruz.Belirlediğimiz işlemler session ile başlayacak.
session = tf.Session(graph=graph) #Graflar bildiğimiz gibi işlemleri tanımlamak için kullanılıyordu ama bu işlemler sadece bir Session(oturum) içerisinde run edilir. 

# First step is always to initialize all variables. 
# We don't care about the return value, though. It's None.
_ = session.run([init]) # başlangıç değerini göz önünde bulundurmamak için böyle bir işlem yapıyoruz.

for i in range(2000):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 100 == 0:
        print("Loss: ", loss_value)
# test datalarımızı yükledik.
test_images, test_labels = data_yükle(test_veri)
# eğitim datalarını 32x32 değiştirdiğimizden dolayı test datamızıda değiştirdik.
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]

images_ve_labels_goster(test_images32, test_labels) #test datalarımızı gösterdik
predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]
# kaç tane eşleşme oldugunu saydık ve topladık.
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels) #oranı hesapladık.
print("Accuracy: {:.3f}".format(accuracy))
session.close()
