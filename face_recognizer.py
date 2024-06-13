import numpy as np
import cv2 as cv



haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Robert Downey', 'Tom Holland', 'Chris Hemsworth']
# features = np.load('features.npy')
# lables = np.load('lables.npy')


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


img = cv.imread(r'C:\python\opencv\face_recognization\Images\Tom Holland\tom.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.namedWindow("show", cv.WINDOW_NORMAL)
cv.resizeWindow("show", 300,300)
cv.imshow("show", gray)


#Detect the face

face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)


for(x,y,w,h) in face_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    # print(f'Label = {people[label]} with a confidence of {confidence}')


    cv.putText(img, str(people[label]),  (10,28), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=3)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)


cv.namedWindow("Detected", cv.WINDOW_NORMAL)
cv.resizeWindow("Detected", 300, 300)
cv.imshow("Detected", img)

cv.waitKey(0)
