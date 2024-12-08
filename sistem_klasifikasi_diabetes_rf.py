import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Memuat model tanpa normalisasi
with open("model_tanpa_normalisasi_coba.pkl", "rb") as f:
    model_tanpa_normalisasi = pickle.load(f)

# Memuat data normalisasi MinMax
with open("Minmax_coba.pkl", "rb") as f:
    data_minmax = pickle.load(f)

# Memuat data normalisasi Z-Score
with open("ZScore_coba.pkl", "rb") as f:
    data_zscore = pickle.load(f)

# Ekstrak MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(data_minmax["X_train_normalized"])

# Ekstrak StandardScaler
zscore = StandardScaler()
zscore.fit(data_zscore["X_train_normalized"])

# Fungsi prediksi
def predict(model, data):
    prediction = model.predict(data)
    return prediction[0]

# Streamlit UI
st.title("Prediksi Diabetes")

st.header("Krisdova Rio Alvonsa 210411100165")

# Input data
st.header("Masukkan Data")
col1, col2 = st.columns(2)

with col1:
    jeniskelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur = st.number_input("Umur", min_value=0, step=1)

with col2:
    hba1c = st.number_input("HbA1c", min_value=0.0, step=0.1)
    guladarah = st.number_input("Gula Darah", min_value=0.0, step=0.1)

# Konversi jenis kelamin menjadi angka
jeniskelamin_numerik = 0 if jeniskelamin == "Perempuan" else 1

# Data baru
new_data = np.array([[jeniskelamin_numerik, umur, hba1c, guladarah]])

# Pilih model
st.header("Pilih Model")
pilih_model = st.selectbox(
    "Pilih model yang akan digunakan:",
    ("Tanpa Normalisasi", "Normalisasi MinMax", "Normalisasi Z-Score")
)

# Tombol prediksi
if st.button("Prediksi"):
    if pilih_model == "Tanpa Normalisasi":
        result = predict(model_tanpa_normalisasi, new_data)
    elif pilih_model == "Normalisasi MinMax":
        normalized_data = minmax.transform(new_data)
        result = predict(model_tanpa_normalisasi, normalized_data)
    elif pilih_model == "Normalisasi Z-Score":
        normalized_data = zscore.transform(new_data)
        result = predict(model_tanpa_normalisasi, normalized_data)

    # Tampilkan hasil
    st.header("Hasil Prediksi")
    if result == 0:
        st.success("Pasien diprediksi memiliki diabetes.")
    else:
        st.success("Pasien diprediksi tidak memiliki diabetes.")
