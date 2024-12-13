import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os
import cv2

root_dir = os.path.dirname(os.path.abspath(__file__))
models_path=os.path.join(root_dir,'models')

# load trained model
with open(f'{models_path}/cnn_model.pkl', 'rb') as f:
    model = pickle.load(f)


st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to detect if it's Normal or Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    
    # preprocess the image
    st.write("Processing the image...")
    img = np.array(image)
    img = cv2.resize(img, (150, 150))  # resize to match model's input size
    img = img / 255.0  # normalize pixel values
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # predict using the model
    prediction = model.predict(img)
    st.write("Prediction Probability: {:.2f}%".format(prediction[0][0] * 100))

 # display result
    if prediction[0][0] > 0.5:
        st.write("The X-ray indicates **Pneumonia**.")
    else:
        st.write("The X-ray indicates **Normal**.")

# footer
st.write("Model trained using CNN architecture on chest X-ray data.")