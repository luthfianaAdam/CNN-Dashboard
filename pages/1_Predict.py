import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("PREDICT IMAGE USING CNN MODEL")

# Use st.cache_resource to load the model once
@st.cache_resource
def load_my_model():
    # Replace 'path/to/your/model.h5' with your actual model file path
    model = tf.keras.models.load_model('best-cnn-model.h5')
    return model

model = load_my_model()

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image as bytes
    # To read image file buffer as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # You can directly pass the uploaded file object to st.image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    st.write("Image Uploaded Successfully")
    st.write(uploaded_file)


    # Example preprocessing for a model expecting 224x224 RGB images normalized to [0, 1]
    img = Image.open(uploaded_file)
    processed_image = img.convert('RGB')
    processed_image = processed_image.resize((224, 224))
    image_array = np.array(processed_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)

    # Interpret results (example for a binary or multi-class model)
    # NOTE: This part depends heavily on how your model outputs predictions
    # For a simple binary model (0 or 1):
    if prediction > 0.5:
        st.success(f"Prediction: Class 1 (Confidence: {prediction[0][0]:.2f})")
    else:
        st.success(f"Prediction: Class 0 (Confidence: {1 - prediction[0][0]:.2f})")

    # st.write("Model Prediction = Dog")

    # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # st.write(file_bytes.shape)
    # st.write(file_bytes)

