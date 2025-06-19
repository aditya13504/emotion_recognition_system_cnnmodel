# Emotion Recognition System

A real-time facial emotion recognition system using Convolutional Neural Networks (CNN) that can detect and classify emotions such as happy, sad, angry, surprised, fearful, and disgusted.

## Features

- Real-time emotion detection using webcam
- Pre-trained CNN model for accurate emotion classification
- Web interface built with Flask
- Support for both image upload and live detection

## Technologies Used

- Python
- TensorFlow
- OpenCV
- Flask
- HTML/CSS/JavaScript

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```
python app.py
```

Then open your browser and navigate to `http://localhost:5000` to use the application.

## Dataset

The model was trained on a dataset of facial expressions that include the following emotions:
- Angry
- Disgusted
- Fearful
- Happy
- Sad
- Surprised

## Note

The train and test image datasets are not included in this repository due to size constraints. You can download similar datasets from:
- [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [CK+](https://www.kaggle.com/datasets/shawon10/ckplus)
