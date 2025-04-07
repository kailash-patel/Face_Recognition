# Face_Recognition
A real-time face detection and recognition system using dlib and OpenCV with MySQL database integration for logging face detection events.

Overview
This system provides real-time face recognition capabilities using a webcam. It detects faces, compares them against a trained model of known faces, and logs recognition events to a MySQL database. The system consists of:

Face detection and recognition module
Training module for creating face recognition models
Database integration for event logging
Real-time webcam interface

Features

Real-time face detection and recognition from webcam feed
Training interface to create face recognition models from sample images
Configurable recognition tolerance levels
Database logging of recognized and unrecognized faces
Time-based logging controls to prevent duplicate entries
