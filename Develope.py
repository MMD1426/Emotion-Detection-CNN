import cv2
import numpy as np
from keras.models import load_model


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    

labels = ['Angry','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']

model = load_model(r'D:\Project-VSCode\emotion_idenfity\model.h5')

cascade = cv2.CascadeClassifier(r'D:\Project-VSCode\emotion_idenfity\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,600)

while True:

    ret,frame = cap.read()
    frame = cv2.resize(frame,(600,600))

    face_detection = cascade.detectMultiScale(frame,1.3, 15)

    for (x,y,w,h) in face_detection:
        draw_border(frame,(x,y),(x+w,y+h),(0,255,0),2,15,5)
        face = frame[y:y+h,x:x+w]
        face = cv2.resize(face, (128, 128))
        pre_face = face / 255
        pre_face = np.expand_dims(pre_face, axis=0)
        pred = model.predict(pre_face)
        argmax_pred = np.argmax(pred,axis=1)
        label = labels[argmax_pred[0]]
        cv2.putText(frame,label,(x+25,y-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Project',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()