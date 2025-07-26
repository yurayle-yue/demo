import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from PIL import Image
import hashlib
import datetime

# =====================================================================================
# 1. KONFIGURASI & SETUP
# =====================================================================================

st.set_page_config(
    page_title="Nutrisense: Solusi Cerdas",
    page_icon="🥗",
    layout="centered"
)

# Lokasi file & Inisialisasi folder
USER_DATA_FILE = "data/users.json"
SCAN_HISTORY_DIR = "data/scan_history_images/"
for folder in ["models", "data", SCAN_HISTORY_DIR]:
    os.makedirs(folder, exist_ok=True)
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump({}, f)

# Database Gizi
NUTRITION_DB = {
    'apple': {'Kalori': 52, 'Karbohidrat (g)': 14, 'Protein (g)': 0.3, 'Lemak (g)': 0.2},
    'banana': {'Kalori': 89, 'Karbohidrat (g)': 23, 'Protein (g)': 1.1, 'Lemak (g)': 0.3},
    'pizza': {'Kalori': 266, 'Karbohidrat (g)': 33, 'Protein (g)': 11, 'Lemak (g)': 10},
    # Tambahkan item makanan lain sesuai kelas model Anda
}

# =====================================================================================
# 2. MANAJEMEN PENGGUNA
# =====================================================================================
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def load_users():
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users_data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

# =====================================================================================
# 3. PEMUATAN MODEL
# =====================================================================================

# PENTING: Ganti daftar ini dengan nama-nama kelas dari model TFLite Anda,
# pastikan urutannya benar sesuai saat training.
FOOD_CLASSES = ['apple', 'banana', 'pizza'] 

@st.cache_resource
def load_tflite_model(model_path):
    """Memuat model TensorFlow Lite (.tflite)."""
    if not os.path.exists(model_path):
        st.error(f"Model TFLite tidak ditemukan di: {model_path}")
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Gagal memuat model TFLite. Error: {e}")
        return None

@st.cache_resource
def load_h5_model(model_path):
    """Memuat model Keras (.h5)."""
    if not os.path.exists(model_path):
        st.error(f"Model H5 tidak ditemukan di: {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model H5. Error: {e}")
        return None

# =====================================================================================
# 4. HALAMAN-HALAMAN APLIKASI
# =====================================================================================

def main_dashboard():
    st.set_page_config(layout="wide")
    st.sidebar.success(f"Login sebagai: {st.session_state['email']}")
    page = st.sidebar.radio("Menu Utama", ["🏠 Klasifikasi Makanan", "🔬 Prediksi Risiko Penyakit", "⚙️ Pengaturan"])
    
    if page == "🏠 Klasifikasi Makanan":
        food_classification_page()
    elif page == "🔬 Prediksi Risiko Penyakit":
        disease_prediction_page()
    elif page == "⚙️ Pengaturan":
        settings_page()
        
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = None
        st.rerun()

def food_classification_page():
    st.title("🍕 Klasifikasi Makanan & Info Gizi")
    st.write("Unggah gambar satu jenis makanan untuk mengetahui nama dan kandungan gizinya.")

    image_file = st.file_uploader("Pilih file gambar...", type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        interpreter = load_tflite_model('models/bestmodel.tflite')
        if interpreter is None:
            return

        image = Image.open(image_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        with col2:
            with st.spinner("Mengklasifikasi gambar..."):
                # Dapatkan detail input dari model TFLite
                input_details = interpreter.get_input_details()
                input_shape = input_details[0]['shape']
                height, width = input_shape[1], input_shape[2]

                # Pre-processing gambar
                img_resized = image.resize((width, height))
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Normalisasi jika model Anda dilatih dengan cara ini

                # Lakukan prediksi
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()

                # Dapatkan hasil
                output_details = interpreter.get_output_details()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Proses hasil
                predicted_index = np.argmax(output_data)
                confidence = np.max(output_data)
                predicted_class_name = FOOD_CLASSES[predicted_index]

                st.subheader("✅ Hasil Klasifikasi")
                st.success(f"**Prediksi:** {predicted_class_name.capitalize()}")
                st.info(f"**Tingkat Keyakinan:** {confidence:.2%}")

                # Tampilkan info gizi
                nutrition_info = NUTRITION_DB.get(predicted_class_name.lower())
                if nutrition_info:
                    st.subheader("Estimasi Kandungan Gizi")
                    st.dataframe(pd.DataFrame([nutrition_info]))
                else:
                    st.warning("Informasi gizi untuk makanan ini tidak tersedia.")

def disease_prediction_page():
    # Fungsi ini tetap sama seperti sebelumnya
    st.title("🔬 Prediksi Risiko Penyakit Kronis")
    st.write("Prediksi dibuat berdasarkan data kesehatan di profil Anda.")
    user_data = load_users()[st.session_state['email']]
    health_data = user_data.get('health_data')
    if not health_data:
        st.warning("Lengkapi data kesehatan di halaman Pengaturan.")
        return
    st.info("Data kesehatan yang digunakan:"); st.json(health_data)
    if st.button("Jalankan Prediksi Risiko", type="primary"):
        model = load_h5_model('models/disease-prediction-tf-model.h5')
        if model is None: return
        height = health_data['Tinggi Badan (cm)']; weight = health_data['Berat Badan (kg)']
        gender_binary = 1 if health_data['Jenis Kelamin'] == 'Laki-laki' else 0
        age = health_data['Usia']; bp = health_data['Tekanan Darah Sistolik (mmHg)']
        chol = health_data['Kolesterol Total (mg/dL)']; glucose = health_data['Gula Darah Puasa (mg/dL)']
        bmi = weight / ((height / 100) ** 2)
        features = np.array([[height, weight, gender_binary, age, bp, chol, glucose, bmi, 0,0,0,0,0,0,0]]) # Sesuaikan fitur jika perlu
        predictions = model.predict(features)
        st.subheader("Potensi Risiko Penyakit:")
        diseases = ['Anemia', 'Kolesterol Tinggi', 'Gagal Ginjal Kronis', 'Diabetes', 'Penyakit Jantung', 'Hipertensi', 'Sindrom Metabolik', 'Perlemakan Hati', 'Obesitas', 'Stroke']
        high_risk = [f"• **{diseases[i]}** ({predictions[0][i]*100:.1f}%)" for i, prob in enumerate(predictions[0]) if prob > 0.5]
        if high_risk:
            for risk in high_risk: st.error(risk)
        else: st.success("✅ Tidak ada risiko penyakit kronis yang signifikan terdeteksi.")

def settings_page():
    # Fungsi ini tetap sama seperti sebelumnya
    st.title("⚙️ Pengaturan Profil dan Kesehatan")
    users = load_users(); user_data = users[st.session_state['email']]
    with st.form("settings_form"):
        st.subheader("Data Pribadi"); name = st.text_input("Nama", value=user_data.get('name', ''))
        st.subheader("Data Kesehatan")
        health_data = user_data.get('health_data', {})
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            height = h_col1.number_input('Tinggi Badan (cm)', 100, value=health_data.get('Tinggi Badan (cm)', 160))
            weight = h_col1.number_input('Berat Badan (kg)', 30, value=health_data.get('Berat Badan (kg)', 60))
            gender = h_col1.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'], index=0 if health_data.get('Jenis Kelamin', 'Laki-laki') == 'Laki-laki' else 1)
        with h_col2:
            age = h_col2.number_input('Usia', 1, value=health_data.get('Usia', 25))
            bp = h_col2.number_input('Tekanan Darah Sistolik', 70, value=health_data.get('Tekanan Darah Sistolik (mmHg)', 120))
            chol = h_col2.number_input('Kolesterol Total', 100, value=health_data.get('Kolesterol Total (mg/dL)', 180))
            glucose = h_col2.number_input('Gula Darah Puasa', 50, value=health_data.get('Gula Darah Puasa (mg/dL)', 90))
        if st.form_submit_button("Simpan Perubahan", type="primary"):
            users[st.session_state['email']]['name'] = name
            users[st.session_state['email']]['health_data'] = {'Tinggi Badan (cm)': height, 'Berat Badan (kg)': weight, 'Jenis Kelamin': gender, 'Usia': age, 'Tekanan Darah Sistolik (mmHg)': bp, 'Kolesterol Total (mg/dL)': chol, 'Gula Darah Puasa (mg/dL)': glucose}
            save_users(users); st.success("Data berhasil diperbarui!")

def initial_health_data_entry():
    # Fungsi ini tetap sama seperti sebelumnya
    st.title("Satu Langkah Lagi!"); st.header("Masukkan Data Kesehatan Awal Anda")
    with st.form("initial_health_form"):
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            height=h_col1.number_input('Tinggi Badan (cm)',160); weight=h_col1.number_input('Berat Badan (kg)',60); gender=h_col1.selectbox('Jenis Kelamin',['Laki-laki','Perempuan'])
        with h_col2:
            age=h_col2.number_input('Usia',25); bp=h_col2.number_input('Tekanan Darah Sistolik',120); chol=h_col2.number_input('Kolesterol Total',180); glucose=h_col2.number_input('Gula Darah Puasa',90)
        if st.form_submit_button("Simpan dan Lanjutkan", type="primary"):
            health_data = {'Tinggi Badan (cm)': height, 'Berat Badan (kg)': weight, 'Jenis Kelamin': gender, 'Usia': age, 'Tekanan Darah Sistolik (mmHg)': bp, 'Kolesterol Total (mg/dL)': chol, 'Gula Darah Puasa (mg/dL)': glucose}
            users = load_users(); users[st.session_state['email']]['health_data'] = health_data
            save_users(users); st.rerun()

def auth_page():
    # Fungsi ini tetap sama seperti sebelumnya
    st.title("Selamat Datang di Nutrisense 🥗")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email"); password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                users = load_users()
                if email in users and users[email]['password'] == hash_password(password):
                    st.session_state.logged_in = True; st.session_state.email = email; st.session_state.name = users[email].get('name', email)
                    st.rerun()
                else: st.error("Email atau password salah.")
    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Nama Lengkap"); email = st.text_input("Email Baru"); password = st.text_input("Buat Password", type="password"); confirm_password = st.text_input("Konfirmasi Password", type="password")
            if st.form_submit_button("Register", type="primary"):
                users = load_users()
                if email in users: st.error("Email sudah terdaftar.")
                elif password != confirm_password: st.error("Password tidak cocok.")
                elif len(password) < 6: st.error("Password minimal 6 karakter.")
                else:
                    users[email] = {"password": hash_password(password), "name": name, "health_data": None}
                    save_users(users); st.success("Registrasi berhasil! Silakan login.")

# =====================================================================================
# 5. MAIN APP ROUTER
# =====================================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    auth_page()
else:
    if load_users().get(st.session_state.email, {}).get("health_data") is None:
        initial_health_data_entry()
    else:
        main_dashboard()
