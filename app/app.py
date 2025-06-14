import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("app/rf_model.pkl")

# Daftar kolom yang digunakan saat training
model_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
    'Gender_Female', 'Gender_Male', 'CALC_Always', 'CALC_Frequently',
    'CALC_Sometimes', 'CALC_no', 'FAVC_no', 'FAVC_yes', 'SCC_no', 'SCC_yes',
    'SMOKE_no', 'SMOKE_yes', 'family_history_with_overweight_no',
    'family_history_with_overweight_yes', 'CAEC_Always', 'CAEC_Frequently',
    'CAEC_Sometimes', 'CAEC_no', 'MTRANS_Automobile', 'MTRANS_Bike',
    'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']

# Judul aplikasi
st.title("Klasifikasi Obesitas")

# Form input pengguna
st.header("Masukkan Data Pengguna")
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (m)", value=1.70)
weight = st.number_input("Berat Badan (kg)", value=65.0)
fcvc = st.slider("Frekuensi makan sayur (1-3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makanan utama per hari (1-4)", 1.0, 4.0, 3.0)
ch2o = st.slider("Jumlah konsumsi air (1-3)", 1.0, 3.0, 2.0)
faf = st.slider("Frekuensi aktivitas fisik mingguan (0-3)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu layar per hari (0-2)", 0.0, 2.0, 1.0)

gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
calc = st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"])
favc = st.selectbox("Sering konsumsi makanan berkalori tinggi?", ["yes", "no"])
scc = st.selectbox("Konsultasi gizi?", ["yes", "no"])
smoke = st.selectbox("Merokok?", ["yes", "no"])
family_history = st.selectbox("Riwayat keluarga obesitas?", ["yes", "no"])
caec = st.selectbox("Kebiasaan makan di luar?", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi utama", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

if st.button("Prediksi"):
    # Nilai numerik
    input_data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'FCVC': fcvc,
        'NCP': ncp,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue,
        # One-hot encoding manual
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'CALC_Always': 1 if calc == 'Always' else 0,
        'CALC_Frequently': 1 if calc == 'Frequently' else 0,
        'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
        'CALC_no': 1 if calc == 'no' else 0,
        'FAVC_no': 1 if favc == 'no' else 0,
        'FAVC_yes': 1 if favc == 'yes' else 0,
        'SCC_no': 1 if scc == 'no' else 0,
        'SCC_yes': 1 if scc == 'yes' else 0,
        'SMOKE_no': 1 if smoke == 'no' else 0,
        'SMOKE_yes': 1 if smoke == 'yes' else 0,
        'family_history_with_overweight_no': 1 if family_history == 'no' else 0,
        'family_history_with_overweight_yes': 1 if family_history == 'yes' else 0,
        'CAEC_Always': 1 if caec == 'Always' else 0,
        'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
        'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
        'CAEC_no': 1 if caec == 'no' else 0,
        'MTRANS_Automobile': 1 if mtrans == 'Automobile' else 0,
        'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
        'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
        'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
        'MTRANS_Walking': 1 if mtrans == 'Walking' else 0
    }

    # Buat DataFrame dan pastikan urutan kolom sama seperti saat training
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_columns]  # urutkan sesuai saat training

    # Prediksi
    prediction = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi Tingkat Obesitas: {prediction}")
