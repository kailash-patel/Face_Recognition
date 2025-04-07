# Face Recognition System
## Project Overview
Face Recognition System is a real-time face detection and recognition platform with database integration for security applications. The system leverages computer vision and machine learning to identify individuals through webcam feeds and maintain comprehensive access logs.

## Project Structure
### Core Components
- **Face Recognition Module**: Real-time face detection and matching capabilities
- **Training System**: Model creation from sample face images
- **Database Integration**: MySQL logging for recognition events
- **Real-time Interface**: Live webcam monitoring and recognition

### Key Features
- **Real-time Detection**: Instant face detection using webcam feed
- **Face Recognition**: Compare detected faces against known individuals
- **Database Logging**: Record all recognition events with timestamps
- **Configurable Security**: Adjustable recognition tolerance levels
- **Interval Controls**: Prevent duplicate log entries with time-based rules

## Technical Details
### Recognition Implementation
- dlib's facial landmark detection for precise face mapping
- Face encoding using deep neural networks
- Euclidean distance calculations for face matching
- OpenCV integration for image capture and processing

### Database System
- MySQL database for event storage
- Structured logging of recognized and unrecognized faces
- Recognition type classification and timestamp recording
- Queryable history for security auditing

## Getting Started
### Prerequisites
- Python 3.6+
- OpenCV
- dlib
- NumPy
- MySQL Server
- mysql-connector-python

### Installation
1. Clone this repository
   
2. Install Python dependencies:
- pip install opencv-python dlib numpy mysql-connector-python
  
3. Download required model files:
- shape_predictor_68_face_landmarks.dat
- dlib_face_recognition_resnet_model_v1.dat

4. Configure MySQL:
- CREATE DATABASE face_recognition_db;

5. Run the application:
- python main.py

Press 'q' to exit the application


