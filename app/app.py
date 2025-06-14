import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("app/rf_model.pkl")

# Judul aplikasi
st.title("Klasifikasi Obesitas")

# Form input pengguna
st.header("Masukkan Data Pengguna")
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (m)", value=1.70)
weight = st.number_input("Berat Badan (kg)", value=65.0)
fcvc = st.slider("Frekuensi makan sayur (1-3)", 1.0, 3.0, 2.0)
faf = st.slider("Frekuensi aktivitas fisik (0-3)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu penggunaan teknologi (0-2)", 0.0, 2.0, 1.0)
# Tambah fitur lain jika perlu

if st.button("Prediksi"):
    input_data = pd.DataFrame({
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "FCVC": [fcvc],
        "FAF": [faf],
        "TUE": [tue]
    })

    # Prediksi
    pred = model.predict(input_data)[0]
    st.success(f"Prediksi Tingkat Obesitas: {pred}")
