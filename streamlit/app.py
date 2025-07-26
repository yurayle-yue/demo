import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import cv2
from PIL import Image
import hashlib
import datetime

# =====================================================================================
# KONFIGURASI APLIKASI & DATABASE
# =====================================================================================
st.set_page_config(
    page_title="Nutrisense: Solusi Cerdas",
    page_icon="🥗",
    layout="centered" # 'centered' lebih baik untuk form login/register
)

# Lokasi file untuk menyimpan data pengguna (simulasi database)
USER_DATA_FILE = "data/users.json"
SCAN_HISTORY_DIR = "data/scan_history_images/"

# Inisialisasi file & folder jika belum ada
os.makedirs("data", exist_ok=True)
os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump({}, f)

# Contoh Database Gizi
NUTRITION_DB = {
    'apple': {'Kalori': 52, 'Karbohidrat (g)': 14, 'Protein (g)': 0.3, 'Lemak (g)': 0.2},
    'banana': {'Kalori': 89, 'Karbohidrat (g)': 23, 'Protein (g)': 1.1, 'Lemak (g)': 0.3},
    'pizza': {'Kalori': 266, 'Karbohidrat (g)': 33, 'Protein (g)': 11, 'Lemak (g)': 10},
}
FOOD_CLASSES = {0: 'background', 1: 'apple', 2: 'banana', 3: 'pizza'}

# =====================================================================================
# FUNGSI MANAJEMEN PENGGUNA & DATA
# =====================================================================================
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def load_users():
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users_data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

def save_scan_history(email, image_path, detected_foods, nutrition_info):
    users = load_users()
    user_data = users.get(email, {})
    
    if 'scan_history' not in user_data:
        user_data['scan_history'] = []
        
    scan_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "detected_foods": detected_foods,
        "nutrition_info": nutrition_info
    }
    user_data['scan_history'].append(scan_record)
    users[email] = user_data
    save_users(users)

# =====================================================================================
# FUNGSI MEMUAT MODEL (DENGAN CACHING)
# =====================================================================================
# (Fungsi-fungsi load_symptoms_model, load_disease_model, load_food_detection_model,
# dan symptoms_mapping dari jawaban sebelumnya diletakkan di sini untuk kerapian)
# ... (Kode-kode ini sama persis dengan jawaban sebelumnya)

@st.cache_resource
def load_disease_model():
    model_path = 'models/disease-prediction-tf-model.h5'
    if not os.path.exists(model_path): return None
    return tf.keras.models.load_model(model_path, compile=False)

# =====================================================================================
# HALAMAN-HALAMAN APLIKASI
# =====================================================================================

def main_dashboard():
    st.set_page_config(layout="wide") # Ganti layout untuk dashboard
    st.sidebar.success(f"Login sebagai: {st.session_state['email']}")
    
    # Navigasi utama setelah login
    page = st.sidebar.radio("Menu Utama", ["🏠 Dashboard", "🩺 Analisis Gejala", "🔬 Prediksi Risiko Penyakit", "📖 Riwayat Scan", "⚙️ Pengaturan"])

    if page == "🏠 Dashboard":
        display_dashboard()
    elif page == "🩺 Analisis Gejala":
        symptoms_analysis_page()
    elif page == "🔬 Prediksi Risiko Penyakit":
        disease_prediction_page()
    elif page == "📖 Riwayat Scan":
        history_page()
    elif page == "⚙️ Pengaturan":
        settings_page()

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = None
        st.rerun()

def display_dashboard():
    st.title(f"Selamat Datang di Dashboard, {st.session_state['name']}! 👋")
    st.write("Pusat kendali kesehatan dan nutrisi Anda. Apa yang ingin Anda lakukan hari ini?")
    
    st.divider()

    st.header("📸 Analisis Gizi Makanan Anda")
    st.write("Gunakan kamera atau unggah gambar untuk mengetahui kandungan gizi makanan Anda secara instan.")
    
    image_file = st.file_uploader("Unggah gambar makanan...", type=['jpg', 'jpeg', 'png'], key="dashboard_uploader")
    
    # Logika untuk menampilkan hasil scan langsung di dashboard
    if image_file:
        handle_food_scan(image_file)

def handle_food_scan(image_file):
    image = Image.open(image_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar yang dianalisis", use_column_width=True)

    with col2:
        with st.spinner("Menganalisis makanan..."):
            # Placeholder untuk logika prediksi model
            # Ganti dengan logika prediksi Mask R-CNN Anda
            simulated_detected_foods = ['pizza', 'apple'] 
            
            st.subheader("✅ Makanan Terdeteksi & Info Gizi")
            
            all_nutrition_info = {}
            if not simulated_detected_foods:
                st.warning("Tidak ada makanan yang dapat dikenali.")
            else:
                for food_name in simulated_detected_foods:
                    nutrition_info = NUTRITION_DB.get(food_name.lower())
                    if nutrition_info:
                        st.success(f"**{food_name.capitalize()}**")
                        df = pd.DataFrame([nutrition_info])
                        st.dataframe(df)
                        all_nutrition_info[food_name] = nutrition_info
                    else:
                        st.error(f"Info gizi untuk {food_name.capitalize()} tidak ditemukan.")
            
            # Simpan ke riwayat
            img_path = os.path.join(SCAN_HISTORY_DIR, image_file.name)
            image.save(img_path)
            save_scan_history(st.session_state['email'], img_path, simulated_detected_foods, all_nutrition_info)
            st.success("Hasil scan berhasil disimpan ke riwayat Anda!")

def symptoms_analysis_page():
    # Kode dari jawaban sebelumnya untuk halaman ini
    st.title("🩺 Analisis Penyakit Berdasarkan Gejala")
    # ... (Implementasi lengkap dari jawaban sebelumnya)
    st.info("Fitur ini sedang dalam pengembangan.")


def disease_prediction_page():
    st.title("🔬 Prediksi Risiko Penyakit Kronis")
    st.write("Prediksi ini dibuat berdasarkan data kesehatan yang telah Anda simpan di profil.")
    
    users = load_users()
    user_data = users[st.session_state['email']]
    health_data = user_data['health_data']
    
    st.info("Data kesehatan yang digunakan untuk analisis:")
    st.json(health_data)

    if st.button("Jalankan Prediksi Risiko", type="primary"):
        # Logika prediksi dari jawaban sebelumnya
        height = health_data['Tinggi Badan (cm)']
        weight = health_data['Berat Badan (kg)']
        # ... dan seterusnya untuk semua data
        
        # (Placeholder untuk logika kalkulasi fitur & prediksi)
        st.subheader("Potensi Risiko Penyakit:")
        st.error("• **Hipertensi** (75.2%)")
        st.error("• **Kolesterol Tinggi** (68.5%)")
        st.warning("• **Obesitas** (55.1%)")
        st.success("✅ Prediksi selesai dijalankan.")

def history_page():
    st.title("📖 Riwayat Scan Makanan")
    users = load_users()
    history = users[st.session_state['email']].get('scan_history', [])

    if not history:
        st.info("Anda belum memiliki riwayat scan. Coba scan makanan di halaman Dashboard!")
        return

    # Tampilkan dari yang terbaru
    for record in reversed(history):
        with st.expander(f"**Scan pada:** {record['timestamp']}"):
            col1, col2 = st.columns([1,2])
            with col1:
                if os.path.exists(record['image_path']):
                    st.image(record['image_path'])
                else:
                    st.warning("Gambar tidak ditemukan.")
            with col2:
                st.write("**Makanan Terdeteksi:**", ", ".join(record['detected_foods']))
                for food, nutrition in record['nutrition_info'].items():
                    st.write(f"**Gizi {food.capitalize()}:**")
                    st.json(nutrition)


def settings_page():
    st.title("⚙️ Pengaturan Profil dan Kesehatan")
    users = load_users()
    user_data = users[st.session_state['email']]

    with st.form("settings_form"):
        st.subheader("Data Pribadi")
        name = st.text_input("Nama", value=user_data.get('name', ''))
        
        st.subheader("Data Kesehatan")
        health_data = user_data.get('health_data', {})
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            height = h_col1.number_input('Tinggi Badan (cm)', min_value=100, value=health_data.get('Tinggi Badan (cm)', 160))
            weight = h_col1.number_input('Berat Badan (kg)', min_value=30, value=health_data.get('Berat Badan (kg)', 60))
            gender = h_col1.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'], index=0 if health_data.get('Jenis Kelamin', 'Laki-laki') == 'Laki-laki' else 1)
        with h_col2:
            age = h_col2.number_input('Usia', min_value=1, value=health_data.get('Usia', 25))
            bp = h_col2.number_input('Tekanan Darah Sistolik', min_value=70, value=health_data.get('Tekanan Darah Sistolik (mmHg)', 120))
            chol = h_col2.number_input('Kolesterol Total', min_value=100, value=health_data.get('Kolesterol Total (mg/dL)', 180))
            glucose = h_col2.number_input('Gula Darah Puasa', min_value=50, value=health_data.get('Gula Darah Puasa (mg/dL)', 90))

        if st.form_submit_button("Simpan Perubahan", type="primary"):
            users[st.session_state['email']]['name'] = name
            users[st.session_state['email']]['health_data'] = {
                'Tinggi Badan (cm)': height, 'Berat Badan (kg)': weight, 'Jenis Kelamin': gender,
                'Usia': age, 'Tekanan Darah Sistolik (mmHg)': bp, 'Kolesterol Total (mg/dL)': chol,
                'Gula Darah Puasa (mg/dL)': glucose
            }
            save_users(users)
            st.success("Data berhasil diperbarui!")

def initial_health_data_entry():
    st.title("Satu Langkah Lagi!")
    st.header("Masukkan Data Kesehatan Awal Anda")
    st.write("Data ini akan digunakan untuk memberikan prediksi dan rekomendasi yang dipersonalisasi.")
    
    with st.form("initial_health_form"):
        # (Form ini mirip dengan form di halaman settings)
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            height = h_col1.number_input('Tinggi Badan (cm)', min_value=100, value=160)
            weight = h_col1.number_input('Berat Badan (kg)', min_value=30, value=60)
            gender = h_col1.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        with h_col2:
            age = h_col2.number_input('Usia', min_value=1, value=25)
            bp = h_col2.number_input('Tekanan Darah Sistolik', min_value=70, value=120)
            chol = h_col2.number_input('Kolesterol Total', min_value=100, value=180)
            glucose = h_col2.number_input('Gula Darah Puasa', min_value=50, value=90)
            
        submitted = st.form_submit_button("Simpan dan Lanjutkan ke Dashboard", type="primary")
        
        if submitted:
            health_data = {
                'Tinggi Badan (cm)': height, 'Berat Badan (kg)': weight, 'Jenis Kelamin': gender,
                'Usia': age, 'Tekanan Darah Sistolik (mmHg)': bp, 'Kolesterol Total (mg/dL)': chol,
                'Gula Darah Puasa (mg/dL)': glucose
            }
            users = load_users()
            users[st.session_state['email']]['health_data'] = health_data
            save_users(users)
            st.success("Data kesehatan berhasil disimpan!")
            st.rerun() # Muat ulang untuk masuk ke dashboard
            
# =====================================================================================
# HALAMAN LOGIN & REGISTER
# =====================================================================================

def auth_page():
    st.title("Selamat Datang di Nutrisense 🥗")
    st.write("Silakan login untuk melanjutkan atau register jika Anda pengguna baru.")
    
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login", type="primary")
            
            if login_button:
                users = load_users()
                if email in users and users[email]['password'] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    st.session_state.name = users[email].get('name', email)
                    st.rerun()
                else:
                    st.error("Email atau password salah.")

    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Nama Lengkap")
            email = st.text_input("Email Baru")
            password = st.text_input("Buat Password", type="password")
            confirm_password = st.text_input("Konfirmasi Password", type="password")
            register_button = st.form_submit_button("Register", type="primary")

            if register_button:
                users = load_users()
                if email in users:
                    st.error("Email sudah terdaftar.")
                elif password != confirm_password:
                    st.error("Password tidak cocok.")
                elif len(password) < 6:
                    st.error("Password minimal 6 karakter.")
                else:
                    users[email] = {
                        "password": hash_password(password),
                        "name": name,
                        "health_data": None,
                        "scan_history": []
                    }
                    save_users(users)
                    st.success("Registrasi berhasil! Silakan login.")

# =====================================================================================
# MAIN APP ROUTER
# =====================================================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.email = None
    st.session_state.name = None

if not st.session_state.logged_in:
    auth_page()
else:
    users = load_users()
    if users[st.session_state.email].get("health_data") is None:
        initial_health_data_entry()
    else:
        main_dashboard()
