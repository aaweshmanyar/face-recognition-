import numpy as np
import cv2 as cv
import os

# --- Setup ---
haar_path = 'haar_face.xml'
model_path = 'face_trained.yml'
img_path = r'C:\python\opencv\face_recognization\Images\Tom Holland\tom.jpg'
people = ['Robert Downey', 'Tom Holland', 'Chris Hemsworth']

# --- Load Haar Cascade ---
if not os.path.exists(haar_path):
    raise FileNotFoundError("Haar cascade XML file not found.")
haar_cascade = cv.CascadeClassifier(haar_path)

# --- Load Trained Model ---
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model file not found.")
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)

# --- Load Image ---
img = cv.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {img_path}")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# --- Show Grayscale Image ---
cv.namedWindow("Grayscale View", cv.WINDOW_NORMAL)
cv.resizeWindow("Grayscale View", 300, 300)
cv.imshow("Grayscale View", gray)

# --- Detect Faces ---
face_rects = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# --- Predict and Annotate ---
for (x, y, w, h) in face_rects:
    face_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)

    name = people[label] if label < len(people) else "Unknown"
    text = f"{name} ({confidence:.2f})"

    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.putText(img, text, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

# --- Show Final Output ---
cv.namedWindow("Face Recognition Result", cv.WINDOW_NORMAL)
cv.resizeWindow("Face Recognition Result", 300, 300)
cv.imshow("Face Recognition Result", img)

cv.waitKey(0)
cv.destroyAllWindows()
