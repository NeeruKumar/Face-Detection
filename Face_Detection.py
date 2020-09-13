import cv2

face_c=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:
    _,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=face_c.detectMultiScale(gray,1.5,10)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow('face',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
