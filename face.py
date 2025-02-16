import os
import cv2
import dlib
import numpy as np
import pickle
import threading
import time
import traceback

from db import DatabaseManager

class FaceRecognition:
    def __init__(self, 
                 model_path='face_recognition_model.pkl', 
                 tolerance=0.4, 
                 log_interval=300):  # 5 minutes between logs for same face
        """
        Initialize face recognition system
        
        Parameters:
        - model_path: Path to pre-trained model
        - tolerance: Face recognition distance tolerance
        - log_interval: Time in seconds between logging same face
        """
        # Set up directories
        self.models_dir = os.path.join(os.getcwd(), 'models')
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Use full path for shape predictor
        shape_predictor_path = os.path.join(self.models_dir, 'shape_predictor_68_face_landmarks.dat')
        self.sp = dlib.shape_predictor(shape_predictor_path)
        
        # Use full path for face recognition model
        face_rec_model_path = os.path.join(self.models_dir, 'dlib_face_recognition_resnet_model_v1.dat')
        self.face_rec = dlib.face_recognition_model_v1(face_rec_model_path)
        
        # Training data
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        
        # Logging management
        self.last_logged_faces = {}  # Track last logged time for each face
        self.log_interval = log_interval
        self.logging_lock = threading.Lock()
        
        # Load pre-trained model
        self.load_model(os.path.join(self.models_dir, model_path))
    
    def load_model(self, model_path):
        """
        Load trained face recognition data from a pickle file
        
        Parameters:
        - model_path: Path to the model pickle file
        """
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.known_face_encodings = model_data['known_face_encodings']
                    self.known_face_names = model_data['known_face_names']
                print(f"Model loaded from {model_path}")
                return True
            else:
                print(f"Model file {model_path} not found!")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def should_log_face(self, name):
        """
        Determine if a face should be logged based on logging interval
        
        Parameters:
        - name: Name of the detected person
        
        Returns:
        - Boolean indicating whether to log the face
        """
        current_time = time.time()
        
        with self.logging_lock:
            # Check if face has been logged before
            if name not in self.last_logged_faces:
                self.last_logged_faces[name] = current_time
                return True
            
            # Check time since last log
            time_since_last_log = current_time - self.last_logged_faces[name]
            
            # Log if time exceeds interval
            if time_since_last_log >= self.log_interval:
                self.last_logged_faces[name] = current_time
                return True
            
            return False
    
    def recognize_faces(self, frame):
        """
        Recognize faces in a given frame
        
        Parameters:
        - frame: Input image frame
        
        Returns:
        - Frame with recognized faces marked
        """
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector(rgb_frame)
        
        # Process each detected face
        for face in faces:
            # Get face location
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            
            # Get face encoding
            shape = self.sp(rgb_frame, face)
            face_encoding = self.face_rec.compute_face_descriptor(rgb_frame, shape)
            
            # Compare with known faces
            name = "Unknown"

            # Use numpy to calculate distances
            if self.known_face_encodings:
                # Calculate Euclidean distances
                face_distances = [
                    np.linalg.norm(np.array(face_encoding) - np.array(known_encoding)) 
                    for known_encoding in self.known_face_encodings
                ]

                # Find the best match
                best_match_index = np.argmin(face_distances)

                # Check if the distance is within tolerance
                if face_distances[best_match_index] < self.tolerance:
                    name = self.known_face_names[best_match_index]
            
            # Log face detection with interval check
            if self.should_log_face(name):
                self.db_manager.log_face_detection(name)
            
            # Draw rectangle and name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame
    
    def real_time_recognition(self):
        """
        Perform real-time face recognition using webcam
        """
        # Start video capture
        cap = cv2.VideoCapture(0)
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Recognize faces in the frame
            frame_with_recognition = self.recognize_faces(frame)
            
            # Display the frame
            cv2.imshow('Real-Time Face Recognition', frame_with_recognition)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()