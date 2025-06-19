document.addEventListener('DOMContentLoaded', () => {
    // Only run the live feed logic on the live.html page
    const videoElement = document.getElementById('videoElement');
    if (!videoElement) {
        console.log("Not on the live feed page.");
        return; // Exit if not on the live page
    }

    const videoContainer = document.getElementById('video-container');
    const socket = io(); // Connect to Socket.IO server (defaults to the same host/port)
    let stream = null;
    let intervalId = null;
    let isProcessing = false; // Flag to prevent sending new frames while one is processing
    const FRAME_RATE = 10; // Send frames 10 times per second (adjust as needed)

    console.log('Attempting to start video stream...');

    // --- Webcam Setup ---
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(mediaStream => {
            stream = mediaStream;
            videoElement.srcObject = stream;
            videoElement.play();
            console.log('Video stream started.');

            // Wait for video metadata to load to get correct dimensions
            videoElement.onloadedmetadata = () => {
                console.log('Video metadata loaded.');
                // Start sending frames only after video is ready
                intervalId = setInterval(sendFrame, 1000 / FRAME_RATE);
            };
        })
        .catch(err => {
            console.error("Error accessing webcam: ", err);
            alert("Could not access webcam. Please ensure permissions are granted and no other application is using it.");
            // Optionally redirect or display a permanent error message
        });

    // --- Send Video Frames --- 
    function sendFrame() {
        if (!isProcessing && videoElement.readyState >= 3) { // Check if video has data
            isProcessing = true; // Set flag
            const canvas = document.createElement('canvas');
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.8); // Use JPEG for smaller size

            // console.log('Sending frame...');
            socket.emit('image', dataUrl);
        }
    }

    // --- Receive Processed Data --- 
    socket.on('processed_image', (data) => {
        // console.log('Received processed data:', data);
        drawResults(data);
        isProcessing = false; // Reset flag after receiving response
    });

    // --- Draw Bounding Boxes and Labels --- 
    function drawResults(results) {
        // Clear previous boxes and labels
        const existingBoxes = videoContainer.querySelectorAll('.bbox, .emotion-label');
        existingBoxes.forEach(box => box.remove());

        if (!results || results.length === 0) return;

        const videoRect = videoElement.getBoundingClientRect();
        const containerRect = videoContainer.getBoundingClientRect();

        results.forEach(result => {
            const { box, emotion, confidence } = result;
            if (!box) return;

            // Scale coordinates
            const scaleX = videoElement.videoWidth > 0 ? videoRect.width / videoElement.videoWidth : 1;
            const scaleY = videoElement.videoHeight > 0 ? videoRect.height / videoElement.videoHeight : 1;

            const div = document.createElement('div');
            // Assign class based on emotion for color-coding
            const emotionClass = emotion ? emotion.toLowerCase() : 'unknown';
            div.className = `bbox ${emotionClass}`; // Add emotion class
            div.style.left = `${box.x * scaleX}px`;
            div.style.top = `${box.y * scaleY}px`;
            div.style.width = `${box.w * scaleX}px`;
            div.style.height = `${box.h * scaleY}px`;

            const label = document.createElement('div');
            label.className = 'emotion-label';

            // --- Simple Emoji Mapping --- 
            const emotionEmojis = {
                angry: 'ANGRY 😠',
                disgusted: 'DISGUSTED 🤢',
                fearful: 'FEARFUL 😨',
                happy: 'HAPPY 😄',
                sad: 'SAD 😢',
                surprised: 'SURPRISED 😮'
            };
            const emoji = emotionEmojis[emotionClass] || '❓'; // Fallback emoji
            // --- End Emoji Mapping --- 

            // Display emoji and confidence percentage
            const confidencePercentage = confidence ? (confidence * 100).toFixed(1) : 0;
            // label.innerText = `${emotion} (${confidencePercentage}%)`; // Old text label
            label.innerHTML = `${emoji} <span style="font-size: 0.7em; vertical-align: middle;">(${confidencePercentage}%)</span>`; // Use emoji + smaller confidence

            label.style.left = `${box.x * scaleX}px`;
            // Position label slightly above the box, preventing it from going off-screen top
            label.style.top = `${Math.max(0, (box.y * scaleY) - 25)}px`; // Adjusted offset

            // Adjust opacity based on confidence (e.g., less confident is more transparent)
            // Make fully opaque above 75%, fade down to 50% opacity at 25% confidence
            const minConfidence = 0.25;
            const maxConfidence = 0.75;
            label.style.opacity = confidence ? Math.min(1, Math.max(0.5, (confidence - minConfidence) / (maxConfidence - minConfidence))) : 0.5;

            videoContainer.appendChild(div);
            videoContainer.appendChild(label);
        });
    }

    // --- Handle Quit --- 
    document.addEventListener('keydown', (event) => {
        if (event.key === 'q' || event.key === 'Q') {
            console.log("'q' pressed, stopping stream and redirecting...");
            stopStreamAndDisconnect();
            // Trigger fade out before redirecting
            document.body.classList.add('fade-out');
            setTimeout(() => {
                window.location.href = '/end'; // Redirect after fade-out
            }, 400); // Match CSS transition duration
        }
    });

    // --- Cleanup Function --- 
    function stopStreamAndDisconnect() {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            videoElement.srcObject = null;
            console.log('Video stream stopped.');
        }
        if (socket && socket.connected) {
            socket.disconnect();
            console.log('Socket disconnected.');
        }
        isProcessing = false;

        // Clear any remaining bounding boxes
        const existingBoxes = videoContainer.querySelectorAll('.bbox, .emotion-label');
        existingBoxes.forEach(box => box.remove());
    }

    // --- Handle Socket.IO connection errors ---
    socket.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
        alert('Could not connect to the server. Please ensure the backend is running.');
        stopStreamAndDisconnect();
    });

    // Optional: Cleanup when the page is unloaded (e.g., browser back button)
    window.addEventListener('beforeunload', () => {
        stopStreamAndDisconnect();
    });

}); 