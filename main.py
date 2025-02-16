from face import FaceRecognition
from db import DatabaseManager

def main():
    # Create database manager to show recent logs (optional)
    db_manager = DatabaseManager()
    
    # Retrieve and display recent logs
    # recent_logs = db_manager.get_recent_logs()
    # print("Recent Face Recognition Logs:")
    # for log in recent_logs:
    #     print(f"{log['timestamp']} - {log['name']} ({log['recognition_type']})")
    
    # Create face recognition instance with pre-trained model
    face_rec = FaceRecognition()
    
    # Start real-time recognition
    face_rec.real_time_recognition()

if __name__ == "__main__":
    main()