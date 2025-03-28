# Face Recognition Application

This application uses your webcam to perform real-time face recognition with advanced features including emotion detection, age estimation, and personal information display. It combines facial recognition with detailed analysis to provide comprehensive information about detected faces.

## Prerequisites

- Python 3.6 or higher
- Webcam
- Required Python packages (installed via requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/praveensharmajava/face-recogination.git
cd face-recogination
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Setup

### Known Faces
1. Create a directory named `known_faces` (it will be created automatically when you run the program)
2. Add photos of people you want to recognize to the `known_faces` directory
3. Name each photo file with the person's name (e.g., `john.jpg`, `jane.png`)
   - Supported formats: .jpg, .jpeg, .png
   - Each image should contain only one face
   - The filename (without extension) will be used as the person's name in recognition

### Personal Information
1. Create an Excel file named `personal_info.xlsx` with the following columns:
   - Name (must match the filename in known_faces)
   - Date_of_Birth
   - LinkedIn_URL
   - Phone_Number
   - Email_ID

## Usage

1. Run the application:
```bash
python face_recognition_app.py
```

2. The application will:
   - Load known faces from the `known_faces` directory
   - Load personal information from `personal_info.xlsx`
   - Open your webcam
   - Start detecting and analyzing faces in real-time

3. Controls:
   - Press 'q' to quit the application

## Features

### Core Features
- Real-time face detection and recognition
- Multiple face tracking
- Personal information display for known faces
- Unknown faces labeled as "Unknown"

### Advanced Analysis
1. Age and Gender Detection
   - Estimates the age of detected faces
   - Determines gender with confidence score

2. Emotion Recognition
   - Detects primary emotions:
     - Happy, Sad, Angry, Neutral
     - Surprised, Fearful, Disgusted

3. Face Direction Tracking
   - Tracks head orientation:
     - Up, Down, Left, Right, Center
   - Provides real-time direction feedback

4. Distance Estimation
   - Estimates approximate distance from camera:
     - Very Close, Close, Medium, Far
   - Based on face size in frame

5. Eye State Detection
   - Monitors eye state (Open/Closed)
   - Can detect blinking

### Display Features
- Black frame around detected faces
- White text on black background for better visibility
- Multi-line information display showing:
  - Name (if recognized)
  - Age and Gender
  - Current Emotion
  - Face Direction
  - Distance from Camera
  - Eye State
  - Personal Information (for known faces)
- Total face count display

## Performance Notes

- Frames are processed at 1/4 resolution for better performance
- Good lighting improves recognition accuracy
- Clear, front-facing photos work best for known face recognition
- The application uses multi-threading for smooth performance
- Real-time analysis may vary based on system capabilities

## Privacy and Security

- Personal information is stored locally in Excel format
- No data is transmitted to external servers
- All processing is done on your local machine
- Recommended to keep personal_info.xlsx secure

## Troubleshooting

- Ensure proper lighting for accurate detection
- Keep face centered and unobstructed for best results
- Check webcam permissions if camera doesn't start
- Verify all required packages are installed
- Ensure image files in known_faces are not corrupted 