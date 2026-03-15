import streamlit as st
import numpy as np

st.title("PREDICT IMAGE USING CNN MODEL")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image as bytes
    # To read image file buffer as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # You can directly pass the uploaded file object to st.image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    st.write("Image Uploaded Successfully")
    st.write(uploaded_file)

    st.write("Model Prediction = Dog")

