import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title("PREDICT IMAGE USING CNN MODEL")

label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load Model
loaded_model = load_model('MODEL_CIFAR10_TA2.keras')

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image as bytes
    # To read image file buffer as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # You can directly pass the uploaded file object to st.image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    st.write("Image Uploaded Successfully")
    # st.write(uploaded_file)



    img = image.load_img(uploaded_file, target_size=(32, 32))  # sesuaikan ukuran
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0  # normalisasi (WAJIB kalau training pakai ini)

    img_array = np.expand_dims(img_array, axis=0)  # jadi (1, 32, 32, 3)

    # prediction
    prediction = loaded_model.predict(img_array)
    predicted_class = label[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write("## Hasil Prediksi")
    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f"### Confidence: **{confidence:.2f}**")
