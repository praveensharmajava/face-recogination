# Face Recognition Application

This application uses your webcam to perform real-time face recognition. It can detect faces in the video stream and identify them if they match with known faces stored in the system.

## Prerequisites

- Python 3.6 or higher
- Webcam
- Required Python packages (installed via requirements.txt)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Setup Known Faces

1. Create a directory named `known_faces` (it will be created automatically when you run the program)
2. Add photos of people you want to recognize to the `known_faces` directory
3. Name each photo file with the person's name (e.g., `john.jpg`, `jane.png`)
   - Supported formats: .jpg, .jpeg, .png
   - Each image should contain only one face
   - The filename (without extension) will be used as the person's name in recognition

## Usage

1. Run the application:
```bash
python face_recognition_app.py
```

2. The application will:
   - Load known faces from the `known_faces` directory
   - Open your webcam
   - Start detecting and recognizing faces in real-time
   - Display the video feed with boxes around detected faces and names below them

3. Controls:
   - Press 'q' to quit the application

## Features

- Real-time face detection
- Face recognition against known faces
- Visual feedback with bounding boxes and names
- Unknown faces are labeled as "Unknown"

## Notes

- The application processes frames at 1/4 resolution for better performance
- Make sure you have good lighting for better recognition accuracy
- Each person should have at least one clear, front-facing photo in the known_faces directory 