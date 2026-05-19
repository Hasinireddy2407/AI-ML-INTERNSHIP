# Module 1 - Landmark Detection and Body Measurement

## What This Module Does
Detects all 33 body landmarks on a person and draws skeleton
lines connecting them. Also calculates basic body measurements
from the detected landmark positions.

## Files
- **module1.py** — Real time landmark detection using webcam
- **landmark_detection.py** — Landmark detection on a saved image file

## Features
- Detects all 33 body landmarks as green dots
- Draws skeleton lines connecting all landmarks
- Labels each landmark
- Calculates 3 body measurements:
  - Shoulder Width
  - Hip Width
  - Arm Length
- Saves output image automatically

## Technologies Used
- Python 3.11
- MediaPipe
- OpenCV
## How to Run

Install dependencies:pip install mediapipe==0.10.14 opencv-python==4.8.0.76 protobuf==4.25.3 numpy==1.24.3
## Input and Output
- module1.py  → Opens webcam and detects landmarks in real time
- landmark_detection.py → Takes a saved photo as input and outputs the same photo with landmarks and skeleton drawn on it
## Input
  <img width="1086" height="1448" alt="person1" src="https://github.com/user-attachments/assets/56bcaeab-c238-47b6-96d1-47a9b0c8295d" />
## Output
  <img width="600" height="900" alt="output_module1" src="https://github.com/user-attachments/assets/c3cb9f5f-64a0-46fc-b817-1d634b0c8304" />
