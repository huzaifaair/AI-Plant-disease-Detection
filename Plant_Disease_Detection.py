# Import necessary libraries
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Load the trained model
model = load_model('plant_disease_model.h5')

# Define class names
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly_blight', 'Corn-Common_rust')

# Streamlit App UI
st.title("🌱 Plant Disease Detection")
st.markdown("Upload an image of a plant leaf to detect the disease.")

# File uploader
plant_image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
submit = st.button("Predict Disease")

# Prediction logic
if submit:
    if plant_image is not None:
        # Convert uploaded image into a format OpenCV can use
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the uploaded image
        st.image(opencv_image, channels="BGR", caption="Uploaded Image")
        st.write("Image Shape:", opencv_image.shape)

        # Preprocess the image for the model
        opencv_image = cv2.resize(opencv_image, (256, 256))  # Resize to match model input size
        opencv_image = opencv_image / 255.0  # Normalize pixel values
        opencv_image = np.expand_dims(opencv_image, axis=0)  # Expand dimensions to match model input

        # Predict the disease
        prediction = model.predict(opencv_image)
        confidence = np.max(prediction) * 100
        result = CLASS_NAMES[np.argmax(prediction)]

        # Display the result
        st.success(f"This is a {result.split('-')[0]} leaf with {result.split('-')[1]}.")
        st.write(f"Prediction Confidence: {confidence:.2f}%")
    else:
        st.error("Please upload an image before clicking 'Predict Disease'.")

