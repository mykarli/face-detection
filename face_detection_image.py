import cv2

# Görseli oku
image = cv2.imread("person.jpg")

# Haar cascade sınıflandırıcısını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Görseli gri tonlamaya çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yüzleri algıla
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Yüzlerin etrafına dikdörtgen çiz
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Sonucu göster
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

