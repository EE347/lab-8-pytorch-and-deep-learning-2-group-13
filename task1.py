import cv2
import os
from picamera2 import Picamera2
import numpy as np

# Set up directories
data_directory = 'data'
folders = ['train/0', 'train/1', 'test/0', 'test/1']
for folder in folders:
    os.makedirs(os.path.join(data_directory, folder), exist_ok=True)

# Initialize and configure the camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to save face images
def save_face_images(label, count):
    saved_count = 0
    while saved_count < 60:
        frame = picam2.capture_array()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces[:1]:  # Only take the first detected face
                face_roi = gray_frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (64, 64))

                if saved_count < 50:
                    folder = 'train'
                else:
                    folder = 'test'

                file_path = os.path.join(data_directory, f'{folder}/{label}/{label}_{saved_count}.jpg')
                cv2.imwrite(file_path, face_resized)
                saved_count += 1
                print(f'Saved {file_path}')

        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Capture 60 images for each teammate
print("Capturing images for person 0. Press 'q' to stop.")
save_face_images(0, 0)
print("Capturing images for person 1. Press 'q' to stop.")
save_face_images(1, 0)

# Release resources
picam2.stop()
cv2.destroyAllWindows()