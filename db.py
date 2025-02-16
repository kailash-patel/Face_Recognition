import mysql.connector
from mysql.connector import Error

class DatabaseManager:
    def __init__(self, 
                 host='localhost', 
                 user='root', 
                 password='1234', 
                 database='face_recognition_db'):
        """
        Initialize database connection
        """
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        
        # Modify connection parameters
        self.connection_config = {
            'host': self.host,
            'user': self.user,
            'password': self.password,
            'database': self.database,
            'auth_plugin': 'mysql_native_password'
        }
        
        # Establish connection
        self.connection = None
        self.cursor = None
        
        # Create database and tables
        self.initialize_database()
    
    def initialize_database(self):
        """
        Create database and necessary tables if they don't exist
        """
        try:
            # Connect to MySQL server without specifying database
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                auth_plugin='mysql_native_password'
            )
            cursor = connection.cursor()
            
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.execute(f"USE {self.database}")
            
            # Create face_logs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                recognition_type ENUM('Recognized', 'Unrecognized') NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            connection.commit()
            cursor.close()
            connection.close()
            
            print("Database and tables created successfully.")
        except Error as e:
            print(f"Error initializing database: {e}")
    
    def _get_connection(self):
        """
        Establish a database connection
        """
        try:
            self.connection = mysql.connector.connect(**self.connection_config)
            self.cursor = self.connection.cursor()
            return self.connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return None
    
    def log_face_detection(self, name):
        """
        Log face detection to the database
        """
        try:
            connection = self._get_connection()
            if connection:
                cursor = connection.cursor()
                recognition_type = "Recognized" if name != "Unknown" else "Unrecognized"
                
                # Insert log entry
                query = """
                INSERT INTO face_logs (name, recognition_type) 
                VALUES (%s, %s)
                """
                cursor.execute(query, (name, recognition_type))
                connection.commit()
                
                print(f"Logged: {name} ({recognition_type})")
                
                cursor.close()
                connection.close()
        except Error as e:
            print(f"Database logging error: {e}")
    
    def get_recent_logs(self, limit=10):
        """
        Retrieve recent face logs
        """
        try:
            connection = self._get_connection()
            if connection:
                cursor = connection.cursor(dictionary=True)
                query = """
                SELECT name, recognition_type, timestamp 
                FROM face_logs 
                ORDER BY timestamp DESC 
                LIMIT %s
                """
                cursor.execute(query, (limit,))
                logs = cursor.fetchall()
                cursor.close()
                connection.close()
                return logs
        except Error as e:
            print(f"Error retrieving logs: {e}")
            return []