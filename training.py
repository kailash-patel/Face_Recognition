import os
import dlib
import numpy as np
import pickle

class FaceTrainer:
    def __init__(self, training_dir='faces', model_path='face_recognition_model.pkl'):
        """
        Initialize face training system using dlib
        
        Parameters:
        training_dir (str): Directory containing face images for training
        model_path (str): Path to save trained face recognition model
        """
        # Initialize face detector and landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(
            'models/shape_predictor_68_face_landmarks.dat'
        )
        
        # Initialize face recognition model
        self.face_rec = dlib.face_recognition_model_v1(
            'models/dlib_face_recognition_resnet_model_v1.dat'
        )
        
        # Training data
        self.known_face_encodings = []
        self.known_face_names = []
        self.training_dir = training_dir
        self.model_path = model_path
    
    def load_image_file(self, file_path):
        """
        Load an image file and convert to RGB using OpenCV
        
        Parameters:
        file_path (str): Path to the image file
        
        Returns:
        numpy array: RGB image
        """
        import cv2
        # Read the image with OpenCV (BGR)
        img = cv2.imread(file_path)
        
        # Convert from BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def face_encodings(self, image):
        """
        Get face encodings for faces in an image
        
        Parameters:
        image (numpy array): RGB image
        
        Returns:
        list: Face encodings
        """
        # Detect faces
        faces = self.detector(image)
        
        # List to store face encodings
        encodings = []
        
        for face in faces:
            # Get facial landmarks
            shape = self.sp(image, face)
            
            # Compute face encoding
            encoding = self.face_rec.compute_face_descriptor(image, shape)
            encodings.append(encoding)
        
        return encodings
    
    def train_recognizer(self):
        """
        Train face recognition by processing training images
        """
        # Check if training directory exists
        if not os.path.exists(self.training_dir):
            print(f"Training directory {self.training_dir} does not exist!")
            return False
        
        # Reset existing encodings
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Iterate through subdirectories (each subdirectory is a person)
        for person_name in os.listdir(self.training_dir):
            person_dir = os.path.join(self.training_dir, person_name)
            
            # Skip if not a directory
            if not os.path.isdir(person_dir):
                continue
            
            # Process images for this person
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                
                try:
                    # Load image
                    image = self.load_image_file(image_path)
                    
                    # Get face encodings
                    face_encodings = self.face_encodings(image)
                    
                    # Add encodings for this person
                    for encoding in face_encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(person_name)
                
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        print(f"Trained with {len(self.known_face_names)} images")
        return len(self.known_face_names) > 0
    
    def save_model(self):
        """
        Save trained face recognition data to a pickle file
        """
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'known_face_encodings': self.known_face_encodings,
                    'known_face_names': self.known_face_names
                }, f)
            print(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

def main():
    # Create face trainer instance
    # Expects a 'faces' directory with subdirectories named after people
    # Each subdirectory should contain training images of that person
    trainer = FaceTrainer()
    
    # Train the recognizer
    if trainer.train_recognizer():
        # Save the trained model
        trainer.save_model()
    else:
        print("Training failed. No images processed.")

if __name__ == "__main__":
    main()