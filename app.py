import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(root_dir, 'models')


# Example: Adding a custom activation function
def custom_activation(x):
    return x

model = load_model(
    os.path.join(models_path, 'complete_model.h5'),
    custom_objects={'custom_activation': custom_activation}
)

# function to preprocess the uploaded image
def preprocess_image(uploaded_image, img_dims):
    # Read and resize the image
    image = Image.open(uploaded_image)
    image = image.resize((img_dims, img_dims))  # Resize to img_dims x img_dims
    image = np.array(image)  # Convert to NumPy array

    # Ensure the image has 3 channels (RGB)
    if image.ndim == 2:  # Grayscale image
        image = np.dstack([image, image, image])  # Convert to 3 channels

    # Normalize pixel values to [0, 1]
    image = image.astype('float32') / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)  #Shape becomes (1, img_dims, img_dims, 3)

    return image

# Streamlit app
st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image to detect if it's Normal or Pneumonia.")

uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    processed_image = preprocess_image(uploaded_file, img_dims=150)

    # Add a button to trigger the prediction
    if st.button("Predict"):
        # Prediction
        prediction = model.predict(processed_image)
        st.write("Prediction Probability: {:.2f}%".format(prediction[0][0] * 100))

        # Display result
        if prediction[0][0] > 0.5:
            st.write("The X-ray indicates **Pneumonia**.")
        else:
            st.write("The X-ray indicates **Normal**.")

# footer
st.write("Model trained using CNN architecture on chest X-ray data.")
