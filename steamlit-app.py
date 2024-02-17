import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformer, webrtc_streamer
import tensorflow as tf

# Load the pre-trained model
my_model = tf.keras.models.load_model('./saved_model/')

# Function to detect emotion
def detect_emotion(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Iterate over each detected face
    for x, y, w, h in faces:
        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize and normalize the grayscale image
        resized_face = cv2.resize(face_roi, (48, 48))
        resized_face = np.expand_dims(resized_face, axis=0)
        resized_face = np.expand_dims(resized_face, axis=-1)
        resized_face = resized_face / 255.0

        # Make prediction using the model
        val = my_model.predict(resized_face)
        prediction_value = np.argmax(val[0])

        # Interpret the prediction and display the emotion
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
        status = emotions[prediction_value]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

class EmotionDetectionTransformer(VideoTransformer):
    def transform(self, frame):
        frame_with_emotion = detect_emotion(frame)
        return frame_with_emotion

def main():
    st.title('Emotion Detection')

    webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetectionTransformer)

if __name__ == '__main__':
    main()