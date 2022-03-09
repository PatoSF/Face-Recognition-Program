#Import the libraries
import cv2
import matplotlib.pyplot as plt


datasetPath='/Users/user/Desktop/opencv-master'
#Loading the cascades
facePath = (cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyePath = (cv2.data.haarcascades + 'haarcascade_eye.xml')
smilePath = (cv2.data.haarcascades + 'haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier(facePath)
eye_cascade = cv2.CascadeClassifier(eyePath)
smile_cascade = cv2.CascadeClassifier(smilePath)

font = cv2.FONT_HERSHEY_SIMPLEX 

#Defining a function that will do the detection
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.putText(frame,'MyFace',(x, y), font,fontScale=1,color=(255,70,120),thickness=2)

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 15)

        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),(0, 255, 0), 2)
            cv2.putText(frame,'Eyes',(x + ex, y + ey), 1, 1, (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.14, 40)

        for(sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh),(0, 0, 255), 2)
            cv2.putText(frame,'Smile',(x + sx, y + sy), 1, 1, (0, 0, 255),2)

        cv2.putText(frame,'Number of Faces detected: ' + str(len(faces)),(30, 30), font, 1,(0,0,0),2)
    return frame

#Doing some Face Recognition with the webcam
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)    
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
cap.release()
cv2.destroyAllWindows()
