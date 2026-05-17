# Driver Distraction and Drowsiness Detection System

An AI-powered real-time Driver Monitoring System developed using Computer Vision and Deep Learning techniques to improve road safety. The system continuously monitors driver behavior through a live webcam feed and detects signs of fatigue and distraction such as prolonged eye closure, yawning, head tilt, and gaze deviation.

## Features

- Real-time driver monitoring
- Eye closure detection for drowsiness analysis
- Yawning detection
- Head tilt detection
- Driver distraction detection
- Live webcam processing
- Alert generation system
- Modular implementation for scalability

## Technologies Used

- Python
- OpenCV
- YOLOv8
- Ultralytics
- NumPy
- CVZone

## Project Structure

```bash
DriverDrowsinessDetectionSystem/
│
├── models/
├── distraction/
├── samples/
├── screenshots/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sr-mithraa/Driver-Drowsiness-Detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## How It Works

The system uses a pretrained YOLOv8 model to detect facial features including the eyes, mouth, and face from live video streams.

- Drowsiness is detected through continuous eye closure monitoring
- Yawning is identified using mouth movement analysis
- Head tilt is calculated using eye-angle estimation
- Distraction is detected based on face position and gaze deviation

Alerts are triggered whenever predefined thresholds are exceeded.

## Future Improvements

- Mobile integration
- Night-time monitoring support
- Deep learning-based fatigue prediction
- Cloud-based monitoring dashboard
- Driver analytics and reporting

## Applications

- Smart vehicle safety systems
- AI-powered driver assistance
- Fleet monitoring solutions
- Road safety applications

## Author

**Mithra S R**
```
