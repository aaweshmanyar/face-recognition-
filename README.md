# Face Recognition Project

## Overview

This project demonstrates a face recognition system using Python and OpenCV. The system detects and recognizes faces in images using Haar cascades for face detection and the Local Binary Patterns Histograms (LBPH) face recognizer for face recognition.

## Features

- Detect faces in images using Haar cascades.
- Recognize faces using the LBPH face recognizer.
- Train the face recognizer with images of multiple individuals.
- Save and load the trained model for future use.

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/face-recognition-project.git
   cd face-recognition-project

2. **Install Dependencies**
   Ensure you have Python and OpenCV installed. You can install OpenCV using pip:
   
   ```sh
   pip install opencv-python
   pip install opencv-contrib-python

## Project Structure
 * face_train.py : Script to train the face recognition model.
 * face_recognizer.py: Script to recognize faces in images using the trained model.
 * haar_face.xml: Haar cascade file for face detection.
 * Images/: Directory containing images of individuals for training and testing.
 * features.npy: Saved features from the training images.
 * labels.npy: Saved labels corresponding to the training images.
 * face_trained.yml: Saved trained face recognition model.

## Demo
![Screenshot (109)](https://github.com/aaweshmanyar/face-recognition-/assets/108227269/040eb98a-7e9d-4b62-a669-44d3e9b77adf)


