import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from deepface import DeepFace
from scipy.spatial import distance as dist
import time
import requests
from io import BytesIO
from PIL import Image
import re

class FaceRecognitionApp:
    def __init__(self):
        # Initialize empty lists for known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load personal information from Excel
        self.personal_info = pd.read_excel('personal_info.xlsx')
        
        # Initialize LinkedIn profile pictures cache
        self.linkedin_pictures = {}
        
        # Load known faces from the 'known_faces' directory
        self.load_known_faces()
        
        # Initialize the webcam
        self.video_capture = cv2.VideoCapture(0)
        
        # Initialize trackers
        self.face_trackers = {}
        self.next_face_id = 0
        
        # Constants for eye aspect ratio
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        self.TOTAL = 0
        
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
    
    def get_personal_info(self, name):
        # Find the person's information in the Excel data
        person_info = self.personal_info[self.personal_info['Name'] == name]
        if not person_info.empty:
            return {
                'DOB': person_info['Date_of_Birth'].iloc[0],
                'LinkedIn': person_info['LinkedIn_URL'].iloc[0],
                'Phone': person_info['Phone_Number'].iloc[0],
                'Email': person_info['Email_ID'].iloc[0]
            }
        return None

    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def estimate_face_direction(self, face_landmarks):
        nose = face_landmarks['nose_bridge'][0]
        chin = face_landmarks['chin'][8]
        left_eye = face_landmarks['left_eye'][0]
        right_eye = face_landmarks['right_eye'][0]
        
        # Calculate angles
        dx = chin[0] - nose[0]
        dy = chin[1] - nose[1]
        angle_vertical = np.degrees(np.arctan2(dy, dx))
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_horizontal = np.degrees(np.arctan2(dy, dx))
        
        # Determine face direction
        direction = {
            'vertical': 'center',
            'horizontal': 'center'
        }
        
        if angle_vertical < -20:
            direction['vertical'] = 'up'
        elif angle_vertical > 20:
            direction['vertical'] = 'down'
            
        if angle_horizontal < -10:
            direction['horizontal'] = 'left'
        elif angle_horizontal > 10:
            direction['horizontal'] = 'right'
            
        return direction

    def estimate_face_distance(self, face_location):
        # Calculate face size
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        face_size = (face_width + face_height) / 2
        
        # Rough distance estimation (calibrated for typical webcam)
        # You may need to adjust these values based on your camera
        if face_size > 250:
            return "Very Close"
        elif face_size > 200:
            return "Close"
        elif face_size > 150:
            return "Medium"
        else:
            return "Far"

    def get_linkedin_activity(self, linkedin_url):
        if not linkedin_url or pd.isna(linkedin_url):
            return None
            
        try:
            # Extract LinkedIn profile ID from URL
            profile_id = re.search(r'in/([^/]+)', linkedin_url)
            if not profile_id:
                return None
                
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Construct the profile URL
            profile_url = f"https://www.linkedin.com/in/{profile_id.group(1)}/"
            
            # Get the profile page
            response = requests.get(profile_url, headers=headers)
            print(response.text)
            if response.status_code == 200:
                # Extract recent activity information
                activity_info = []
                
                # Look for recent posts
                posts = re.findall(r'<span class="visually-hidden">([^<]+)</span>', response.text)
                if posts:
                    activity_info.extend(posts[:3])  # Get up to 3 recent posts
                
                # Look for recent connections
                connections = re.findall(r'connected with ([^<]+)', response.text)
                if connections:
                    activity_info.extend(connections[:2])  # Get up to 2 recent connections
                
                # Look for recent endorsements
                endorsements = re.findall(r'endorsed ([^<]+) for', response.text)
                if endorsements:
                    activity_info.extend(endorsements[:2])  # Get up to 2 recent endorsements
                
                return activity_info[:5]  # Return up to 5 recent activities
                
        except Exception as e:
            print(f"Error fetching LinkedIn activity: {e}")
            return None

    def run(self):
        print("Starting Face Recognition App...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Reduce frame size for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)

            # Process each face in the frame
            for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                personal_info = None

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    personal_info = self.get_personal_info(name)

                # Analyze face
                try:
                    face_img = frame[top:bottom, left:right]
                    analysis = DeepFace.analyze(face_img, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                    
                    # Get face direction
                    direction = self.estimate_face_direction(landmarks)
                    
                    # Get face distance
                    distance = self.estimate_face_distance((top, right, bottom, left))
                    
                    # Calculate eye aspect ratio
                    left_eye = landmarks['left_eye']
                    right_eye = landmarks['right_eye']
                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    # Determine if eyes are closed
                    eyes_status = "Open" if ear > self.EYE_AR_THRESH else "Closed"
                    
                    # Draw the results
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
                    
                    # Create info text
                    info_text = [
                        f"Name: {name}",
                        f"Age: {analysis[0]['age']}",
                        f"Gender: {'Male' if analysis[0]['gender'] == 'Man' else 'Female'}",
                        f"Emotion: {analysis[0]['dominant_emotion']}",
                        f"Direction: {direction['horizontal']}-{direction['vertical']}",
                        f"Distance: {distance}",
                        f"Eyes: {eyes_status}"
                    ]

                    # Add personal information if available
                    if personal_info:
                        info_text.extend([
                            f"DOB: {personal_info['DOB']}",
                            f"Phone: {personal_info['Phone']}",
                            f"Email: {personal_info['Email']}"
                        ])
                        
                        # Display LinkedIn activity if available
                        if personal_info['LinkedIn']:
                            print(f"Fetching LinkedIn activity for: {personal_info['LinkedIn']}")  # Debug print
                            linkedin_activity = self.get_linkedin_activity(personal_info['LinkedIn'])
                            if linkedin_activity:
                                print("Successfully loaded LinkedIn activity")  # Debug print
                                # Add activity information to the display
                                info_text.append("Recent LinkedIn Activity:")
                                for activity in linkedin_activity:
                                    info_text.append(f"- {activity}")
                            else:
                                print("Failed to load LinkedIn activity")  # Debug print

                    # Display the information
                    y = top - 10
                    for text in info_text:
                        y -= 30
                        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                except Exception as e:
                    print(f"Error processing face: {e}")

            # Display the resulting frame
            cv2.imshow('Face Recognition', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and close windows
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run() 