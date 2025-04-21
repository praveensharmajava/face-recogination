# Face Recognition Application

A sophisticated face recognition application built with Python that includes features like emotion detection, age estimation, gender detection, and LinkedIn integration.

## Features

- Real-time face detection and recognition
- Emotion analysis
- Age and gender estimation
- Face direction tracking
- Eye state monitoring (open/closed)
- Distance estimation
- Personal information display
- LinkedIn profile integration
- Recent activity tracking

## Requirements

- Python 3.8+
- OpenCV
- face_recognition
- deepface
- numpy
- pandas
- requests
- Pillow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-app.git
cd face-recognition-app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Add known face images to the `known_faces` directory
2. Update personal information in `personal_info.xlsx`
3. Run the application:
```bash
python face_recognition_app.py
```

4. Press 'q' to quit the application

## Configuration

- Personal information is stored in `personal_info.xlsx`
- Known faces should be added to the `known_faces` directory
- Each known face image should be named according to the person's name

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 