"""
This file detects face and eye and draws a spectacle around eyes.
"""

import cv2
import numpy as np 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        try:
            # RIGHT EYE
            cv2.rectangle(roi_color, (int(left_eye[0][0]),int(left_eye[0][1]*1.5)), (left_eye[0][0]+left_eye[0][2], int(left_eye[0][1]*1.5)+int(left_eye[0][3]*0.5)), (139,0,139), 2)
            # LEFT EYE
            cv2.rectangle(roi_color, (left_eye[1][0],int(left_eye[1][1]*1.5)), (left_eye[1][0]+left_eye[1][2], int(left_eye[1][1]*1.5)+int(left_eye[1][3]*0.5)), (139,0,139), 2)

            cv2.line(roi_color, (left_eye[0][0], int(left_eye[0][1]*1.5)), (left_eye[1][0]+left_eye[1][2], int(left_eye[1][1]*1.5)), (139,0,139), 3)
        except:
            continue

        
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
