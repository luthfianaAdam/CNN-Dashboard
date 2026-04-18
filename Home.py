import streamlit as st

st.title("ARTIFICIAL INTELLIGENCE")
st.title("GROUP 5")
st.header("OUR TEAM", divider="gray")
st.subheader("LUTHFIANA ADAM MALDINI (2902724045)")
st.subheader("ARTI SURYANING TYAS (2902724341)")
st.subheader("RATIH DEWI SETYO JATI (2902696956)")
st.subheader("RESKI ROMAITO HUTABARAT (2802647392)")
st.subheader("MUHAMMAD RABBANI SYAWAL (2802635190)")

st.header("MODEL SUMMARY", divider="gray")
st.write("""
         Web ini adalah aplikasi prediksi gambar sederhana yang memanfaatkan Convolutional Neural Network (CNN). 
         Model telah dilatih menggunakan dataset CIFAR-10, yang terdiri dari 10 kelas objek seperti airplane, automobile, cat, dog, dan lainnya. 
         Pengguna dapat mengunggah gambar, kemudian sistem akan memprediksi kelas objek pada gambar tersebut secara otomatis.
         """)

st.image('grafik accuracy dan loss.png', caption='loss and accuracy graph', use_column_width=True)
st.image('score.png', caption='Classification Report', use_column_width=True)
