import streamlit as st
import cv2
import imghdr
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


st.title('Image Classification App')


def predict_image(model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 'Invalid image file'
    img_resized = tf.image.resize(img, (256, 256))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    prediction = model.predict(img_array)
    return 'Dog' if prediction > 0.5 else 'Cat'


uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_path = uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    model = load_model(os.path.join('models', 'imageclassifier.h5'))
    prediction = predict_image(model, image_path)

    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {prediction}')
