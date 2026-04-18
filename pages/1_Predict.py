import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("PREDICT IMAGE USING CNN MODEL")

# model = tf.keras.models.load_model("model.h5")
# model = tf.keras.models.load_model("best-cnn-model.h5")
# Use st.cache_resource to load the model once
# @st.cache_resource
# def load_my_model():
#     # Replace 'path/to/your/model.h5' with your actual model file path
#     model = tf.keras.models.load_model("model.h5")
#     return model

# model = load_my_model()

# Load Model
loaded_model = load_model('MODEL_CIFAR10_TA1.keras')

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image as bytes
    # To read image file buffer as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # You can directly pass the uploaded file object to st.image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    st.write("Image Uploaded Successfully")
    st.write(uploaded_file)


    # tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = loaded_model.predict(img_array)
    # predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write("### Hasil Prediksi")
    # st.write(f"Class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")

