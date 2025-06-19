# Apply monkey patch first
import eventlet
eventlet.monkey_patch() # Patch standard libraries for eventlet compatibility

import base64
import logging
import cv2
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
import tensorflow as tf # Add TensorFlow import
import os # Add os import for path joining

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key!' # Change this in production!
socketio = SocketIO(app, async_mode='eventlet')

# --- Model and Detector Loading ---
MODEL_PATH = 'emotion_model_cnn_scratch.keras'
FACE_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# Define class names in the order TensorFlow's image_dataset_from_directory likely found them
# Verify this order matches the output during training if issues arise
CLASS_NAMES = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']

# Load the trained emotion model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    IMG_HEIGHT = model.input_shape[1]
    IMG_WIDTH = model.input_shape[2]
    logging.info(f"Successfully loaded model from {MODEL_PATH} with input shape {(IMG_HEIGHT, IMG_WIDTH)}")
except Exception as e:
    logging.error(f"Error loading model from {MODEL_PATH}: {e}")
    model = None # Set model to None if loading fails

# Load the face cascade classifier
if os.path.exists(FACE_CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    logging.info(f"Successfully loaded face cascade from {FACE_CASCADE_PATH}")
else:
    logging.error(f"Error loading face cascade: File not found at {FACE_CASCADE_PATH}")
    face_cascade = None

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_app', methods=['POST'])
def start_app():
    username = request.form.get('username')
    if not username:
        return redirect(url_for('index')) # Or show an error
    session['username'] = username
    return render_template('app.html', username=username)

@app.route('/live')
def live():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('live.html')

@app.route('/end')
def end():
    session.pop('username', None) # Clear username on exit
    return render_template('end.html')

@socketio.on('connect')
def handle_connect():
    log.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    log.info('Client disconnected')

@socketio.on('image')
def handle_image(data_url):
    # Ensure model and cascade are loaded
    if model is None or face_cascade is None:
        log.warning("Model or face cascade not loaded. Skipping prediction.")
        # Optionally emit an error to the client
        emit('processing_error', {'error': 'Server model/detector not ready.'})
        return

    try:
        # Decode the base64 image
        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode in color first, as users see color, but we'll process in gray
        frame_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_color is None:
            log.warning("Received empty image frame.")
            return

        # Convert to grayscale for face detection and model prediction
        frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1, # Adjust scaleFactor (e.g., 1.1 to 1.4)
            minNeighbors=5,  # Adjust minNeighbors (e.g., 3 to 6)
            minSize=(30, 30) # Ignore small detections
        )

        processed_data = []
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest) from the grayscale frame
            face_roi_gray = frame_gray[y:y+h, x:x+w]

            # Preprocess the face ROI for the model
            face_resized = cv2.resize(face_roi_gray, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            face_normalized = face_resized / 255.0
            # Reshape for the model: (1, height, width, channels)
            face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)

            # Predict emotion using the loaded model
            prediction = model.predict(face_input)
            pred_index = np.argmax(prediction)
            confidence = prediction[0][pred_index]
            emotion_label = CLASS_NAMES[pred_index]
            log.info(f"Raw prediction: {prediction}, Index: {pred_index}, Label: {emotion_label}, Confidence: {confidence:.4f}")

            # Prepare data structure for the client
            # Send original coordinates (x, y, w, h)
            # Convert numpy int32 types to standard Python int
            processed_data.append({
                'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'emotion': emotion_label,
                'confidence': float(confidence) # Already converting confidence to float
            })

            # Optional: Draw rectangle and label on the *color* frame for debugging/display if needed server-side
            # cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.putText(frame_color, f"{emotion_label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Send results back to the client
        emit('processed_image', processed_data)

    except Exception as e:
        log.error(f"Error processing image: {e}", exc_info=True) # Log full traceback
        # Optionally emit an error event to the client
        emit('processing_error', {'error': 'An internal server error occurred during processing.'})

if __name__ == '__main__':
    log.info("Starting Flask-SocketIO server...")
    # Use 0.0.0.0 to make it accessible on your local network
    # Set debug=False for production generally, but True is fine for development
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 