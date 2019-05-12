import cv2

cc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("women.jpg")

cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cc.detectMultiScale(cinza, scaleFactor = 1.05, minNeighbors=7)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

cv2.imshow("Resultado", img)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
