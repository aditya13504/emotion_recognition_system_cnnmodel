# Emotion Recognition System

## Overview
This project is a real-time emotion recognition system built using a Convolutional Neural Network (CNN) model. The system detects faces in images or video streams and classifies them into one of six emotions: **angry**, **disgusted**, **fearful**, **happy**, **sad**, and **surprised**. It includes a Flask-based web application for user interaction and visualization.

## Features
- **Real-Time Emotion Detection**: Uses OpenCV for face detection and a trained CNN model for emotion classification.
- **Web Interface**: A Flask web application with live video feed support.
- **Custom Model Training**: Includes a script to train the CNN model from scratch using labeled datasets.
- **Interactive Visualizations**: Displays detected emotions and confidence levels in real-time.

## Project Structure
```
.
├── app.py                 # Main Flask application
├── train_model.py         # Script to train the CNN model
├── emotion_model_cnn_scratch.keras  # Pre-trained CNN model
├── requirements.txt       # Python dependencies
├── static/                # Static files (CSS, JS, images)
│   ├── css/
│   │   └── style.css
│   ├── images/
│   │   └── eye.jpg
│   ├── js/
│       ├── background.js
│       ├── index_animations.js
│       └── script.js
├── templates/             # HTML templates
│   ├── app.html
│   ├── end.html
│   ├── index.html
│   └── live.html
├── train/                 # Training dataset (organized by emotion)
├── test/                  # Testing dataset (organized by emotion)
└── README.md              # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/aditya13504/emotion_recognition_system_cnnmodel.git
   cd emotion_recognition_system_cnnmodel
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `train/` and `test/` directories are populated with labeled subdirectories for each emotion.

## Usage

### Running the Web Application
1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`.

3. Interact with the web interface to upload images or use the live video feed for emotion detection.

### Training the Model
1. Modify the `train/` and `test/` directories with your dataset.
2. Run the training script:
   ```bash
   python train_model.py
   ```
3. The trained model will be saved as `emotion_model_cnn_scratch.keras`.

## Dependencies
The project requires the following Python libraries:
- `tensorflow>=2.0.0`
- `opencv-python>=4.5.0`
- `flask>=2.0.0`
- `flask-socketio>=5.0.0`
- `eventlet>=0.30.0`
- `numpy>=1.19.0`
- `pillow>=8.0.0`

Install them using:
```bash
pip install -r requirements.txt
```

## Dataset
The project expects the dataset to be organized as follows:
```
train/
  angry/
  disgusted/
  fearful/
  happy/
  sad/
  surprised/
test/
  angry/
  disgusted/
  fearful/
  happy/
  sad/
  surprised/
```
Each subdirectory should contain images corresponding to the respective emotion.

## Acknowledgments
- **TensorFlow**: For providing the deep learning framework.
- **OpenCV**: For face detection.
- **Flask**: For building the web application.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
