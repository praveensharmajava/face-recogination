import cv2
import face_recognition
import numpy as np
import os

class FaceRecognitionApp:
    def __init__(self):
        # Initialize empty lists for known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load known faces from the 'known_faces' directory
        self.load_known_faces()
        
        # Initialize the webcam
        self.video_capture = cv2.VideoCapture(0)
        
    def load_known_faces(self):
        # Create known_faces directory if it doesn't exist
        if not os.path.exists('known_faces'):
            os.makedirs('known_faces')
            print("Created 'known_faces' directory. Please add face images there.")
            return
            
        # Load all images from known_faces directory
        for filename in os.listdir('known_faces'):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Load the image
                image_path = os.path.join('known_faces', filename)
                face_image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                face_encodings = face_recognition.face_encodings(face_image)
                
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    # Use filename without extension as the person's name
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                    print(f"Loaded face: {name}")
    
    def run(self):
        while True:
            # Capture frame from webcam
            ret, frame = self.video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            for face_encoding in face_encodings:
                # Check if the face matches any known face
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                if self.known_face_encodings:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                
                face_names.append(name)
            
            # Draw results on frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since we scaled down the frame
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw name label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    print("Starting Face Recognition App...")
    print("Press 'q' to quit")
    app.run() 