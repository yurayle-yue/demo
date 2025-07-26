import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import os
import hashlib
import datetime
from PIL import Image

# =====================================================================================
# 1. KONFIGURASI & SETUP
# =====================================================================================

# Atur konfigurasi halaman di awal
# Inisialisasi session state untuk tema jika belum ada
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True # Default ke mode gelap

st.set_page_config(
    page_title="Nutrisense: Solusi Cerdas",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Terapkan tema berdasarkan session state
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .st-emotion-cache-16txtl3 {
            color: #FAFAFA;
        }
        .st-emotion-cache-1avcm0n h1 {
            color: #FAFAFA;
        }
        </style>
    """, unsafe_allow_html=True)
else:
     st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)


# Lokasi file & Inisialisasi folder
USER_DATA_FILE = "data/users.json"
SCAN_HISTORY_DIR = "data/scan_history_images/"
for folder in ["models", "data", SCAN_HISTORY_DIR]:
    os.makedirs(folder, exist_ok=True)
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump({}, f)

# Database Gizi dengan info alergen
NUTRITION_DB = {
    'apple': {'Kalori': 52, 'Karbohidrat (g)': 14, 'Protein (g)': 0.3, 'Lemak (g)': 0.2, 'Alergen': []},
    'banana': {'Kalori': 89, 'Karbohidrat (g)': 23, 'Protein (g)': 1.1, 'Lemak (g)': 0.3, 'Alergen': []},
    'pizza': {'Kalori': 266, 'Karbohidrat (g)': 33, 'Protein (g)': 11, 'Lemak (g)': 10, 'Alergen': ['gluten', 'susu']},
    'nasi': {'Kalori': 130, 'Karbohidrat (g)': 28, 'Protein (g)': 2.7, 'Lemak (g)': 0.3, 'Alergen': []},
    'telur': {'Kalori': 155, 'Karbohidrat (g)': 1.1, 'Protein (g)': 13, 'Lemak (g)': 11, 'Alergen': ['telur']},
}

# =====================================================================================
# 2. MANAJEMEN DATA & PENGGUNA
# =====================================================================================
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def load_users():
    with open(USER_DATA_FILE, 'r') as f:
        return json.load(f)

def save_users(users_data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

def save_scan_history(email, image_path, result):
    users = load_users()
    user_data = users.get(email, {})
    if 'scan_history' not in user_data: user_data['scan_history'] = []
    scan_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path,
        "result": result
    }
    user_data['scan_history'].append(scan_record)
    users[email] = user_data
    save_users(users)

# =====================================================================================
# 3. PEMUATAN MODEL
# =====================================================================================
@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path): return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat {model_path}. Error: {e}")
        return None

# =====================================================================================
# 4. LOGIKA ANALISIS & REKOMENDASI
# =====================================================================================
def get_disease_risks(health_data):
    """Menjalankan model prediksi penyakit dan mengembalikan daftar risiko."""
    model = load_model('models/disease-prediction-tf-model.h5')
    if model is None: return ["Model prediksi risiko tidak tersedia."]

    height = health_data['Tinggi Badan (cm)']; weight = health_data['Berat Badan (kg)']
    gender_binary = 1 if health_data['Jenis Kelamin'] == 'Laki-laki' else 0
    age = health_data['Usia']; bp = health_data['Tekanan Darah Sistolik (mmHg)']
    chol = health_data['Kolesterol Total (mg/dL)']; glucose = health_data['Gula Darah Puasa (mg/dL)']
    bmi = weight / ((height / 100) ** 2)
    bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
    age_category = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
    bp_category = 0 if bp < 120 else 1 if bp < 140 else 2 if bp < 160 else 3
    features = np.array([[height, weight, gender_binary, age, bp, chol, glucose, bmi, bmi_category, age_category, bp_category, bmi * age, bp * age, bmi * bp, weight * 20]])
    
    predictions = model.predict(features)[0]
    diseases = ['Anemia', 'Kolesterol Tinggi', 'Gagal Ginjal Kronis', 'Diabetes', 'Penyakit Jantung', 'Hipertensi', 'Sindrom Metabolik', 'Perlemakan Hati', 'Obesitas', 'Stroke']
    
    return [diseases[i] for i, prob in enumerate(predictions) if prob > 0.5]

def generate_holistic_recommendation(foods, nutrition_summary, health_risks, user_allergies):
    """Membuat rekomendasi berdasarkan semua input."""
    recommendations = []
    warnings = []

    # Aturan Peringatan berdasarkan Risiko Penyakit
    if 'Diabetes' in health_risks or 'Hipertensi' in health_risks:
        if nutrition_summary.get('Karbohidrat (g)', 0) > 30:
            warnings.append("Tinggi Karbohidrat: Porsi perlu diperhatikan untuk menjaga gula darah dan berat badan.")
    if 'Kolesterol Tinggi' in health_risks or 'Penyakit Jantung' in health_risks:
        if nutrition_summary.get('Lemak (g)', 0) > 15:
            warnings.append("Tinggi Lemak: Kurangi konsumsi untuk menjaga kesehatan jantung dan kolesterol.")
    
    # Aturan Peringatan berdasarkan Alergi
    for food_name, nutrition_info in foods.items():
        for allergen in nutrition_info.get('Alergen', []):
            if allergen in user_allergies:
                warnings.append(f"Peringatan Alergi: Makanan ini mengandung '{allergen}', yang merupakan alergen Anda.")

    # Rekomendasi Umum
    if not warnings:
        recommendations.append("Makanan ini tampaknya aman untuk Anda berdasarkan profil kesehatan saat ini. Nikmati secukupnya!")
    if nutrition_summary.get('Protein (g)', 0) > 15:
        recommendations.append("Sumber protein yang baik untuk membangun dan memperbaiki jaringan tubuh.")

    return warnings, recommendations

# =====================================================================================
# 5. UI COMPONENTS (HALAMAN APLIKASI)
# =====================================================================================
def display_holistic_analysis(image, detected_foods):
    user_data = load_users()[st.session_state['email']]
    health_data = user_data.get('health_data')
    user_allergies = [a.strip().lower() for a in user_data.get('allergies', '').split(',')]

    st.subheader("✅ Hasil Analisis Terintegrasi")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Makanan yang Dianalisis", use_column_width=True)
        st.write("**Makanan Terdeteksi:**")
        for food in detected_foods:
            st.success(f"• {food.capitalize()}")

    with col2:
        # 1. Kumpulkan Informasi Gizi
        nutrition_summary = {'Kalori': 0, 'Karbohidrat (g)': 0, 'Protein (g)': 0, 'Lemak (g)': 0}
        all_foods_info = {}
        for food in detected_foods:
            info = NUTRITION_DB.get(food.lower())
            if info:
                all_foods_info[food] = info
                for key in nutrition_summary:
                    nutrition_summary[key] += info.get(key, 0)
        
        st.write("**Estimasi Total Gizi:**")
        st.dataframe(pd.DataFrame([nutrition_summary]))

        # 2. Dapatkan Risiko Penyakit
        with st.spinner("Mengecek profil kesehatan Anda..."):
            health_risks = get_disease_risks(health_data)
        
        # 3. Hasilkan Rekomendasi
        warnings, recommendations = generate_holistic_recommendation(all_foods_info, nutrition_summary, health_risks, user_allergies)

        # 4. Tampilkan Rekomendasi & Peringatan
        st.write("**Rekomendasi & Peringatan Untuk Anda:**")
        if warnings:
            for warning in warnings:
                st.error(f"🚨 {warning}")
        if recommendations:
            for rec in recommendations:
                st.success(f"💡 {rec}")
        
        # Simpan hasil lengkap ke riwayat
        scan_result = {
            "detected_foods": detected_foods,
            "nutrition_summary": nutrition_summary,
            "warnings": warnings,
            "recommendations": recommendations
        }
        img_path = os.path.join(SCAN_HISTORY_DIR, f"{st.session_state['email'].split('@')[0]}_{int(datetime.datetime.now().timestamp())}.png")
        image.save(img_path)
        save_scan_history(st.session_state['email'], img_path, scan_result)
        st.info("Hasil analisis ini telah disimpan ke Riwayat Anda.")

def settings_page():
    st.title("⚙️ Pengaturan")
    users = load_users(); user_data = users[st.session_state['email']]

    # Pengaturan Tema
    st.subheader("Tampilan")
    is_dark = st.toggle("Aktifkan Mode Gelap", value=st.session_state.dark_mode)
    if is_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = is_dark
        st.rerun()

    with st.form("settings_form"):
        st.subheader("Profil Pengguna")
        name = st.text_input("Nama", value=user_data.get('name', ''))
        allergies = st.text_area("Daftar Alergi (pisahkan dengan koma)", value=user_data.get('allergies', ''))

        st.subheader("Data Kesehatan")
        health_data = user_data.get('health_data', {})
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            height = h_col1.number_input('Tinggi (cm)', 100, value=health_data.get('Tinggi Badan (cm)', 160))
            weight = h_col1.number_input('Berat (kg)', 30, value=health_data.get('Berat Badan (kg)', 60))
        with h_col2:
            age = h_col2.number_input('Usia', 1, value=health_data.get('Usia', 25))
            bp = h_col2.number_input('Tekanan Darah Sistolik', 70, value=health_data.get('Tekanan Darah Sistolik (mmHg)', 120))
        
        if st.form_submit_button("Simpan Perubahan", type="primary"):
            users[st.session_state['email']]['name'] = name
            users[st.session_state['email']]['allergies'] = allergies
            users[st.session_state['email']]['health_data'] = {
                'Tinggi Badan (cm)': height, 'Berat Badan (kg)': weight,
                'Usia': age, 'Tekanan Darah Sistolik (mmHg)': bp,
                # Asumsi dari model lama
                'Jenis Kelamin': health_data.get('Jenis Kelamin', 'Laki-laki'),
                'Kolesterol Total (mg/dL)': health_data.get('Kolesterol Total (mg/dL)', 180),
                'Gula Darah Puasa (mg/dL)': health_data.get('Gula Darah Puasa (mg/dL)', 90)
            }
            save_users(users)
            st.success("Data berhasil diperbarui!")

def main_app():
    st.sidebar.title(f"Menu Nutrisense")
    st.sidebar.success(f"Login sebagai: {st.session_state.name}")

    st.sidebar.header("Pusat Analisis")
    if st.sidebar.button("Scan Makanan & Analisis Holistik", type="primary"):
        st.session_state.page = "scan"
    
    st.sidebar.header("Lainnya")
    if st.sidebar.button("Riwayat Analisis"):
        st.session_state.page = "history"
    if st.sidebar.button("Pengaturan Profil"):
        st.session_state.page = "settings"
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.session_state.page = "welcome"
        st.rerun()

    # Konten Halaman Utama
    if st.session_state.page == "scan":
        st.title("📸 Scan Makanan & Analisis Holistik")
        image_file = st.file_uploader("Unggah gambar makanan...", type=['jpg', 'jpeg', 'png'], key="main_uploader")
        if image_file:
            # Placeholder untuk model deteksi, ganti dengan logika model Anda
            simulated_detected_foods = ['nasi', 'telur'] 
            display_holistic_analysis(Image.open(image_file), simulated_detected_foods)

    elif st.session_state.page == "history":
        st.title("📖 Riwayat Analisis")
        history = load_users()[st.session_state['email']].get('scan_history', [])
        if not history:
            st.info("Anda belum memiliki riwayat analisis.")
        for record in reversed(history):
            with st.expander(f"Scan pada: {record['timestamp']}"):
                result = record['result']
                st.write("**Makanan:**", ", ".join(result['detected_foods']))
                st.write("**Gizi:**"); st.json(result['nutrition_summary'])
                st.write("**Peringatan:**")
                for w in result['warnings']: st.error(w)
                st.write("**Rekomendasi:**")
                for r in result['recommendations']: st.success(r)

    elif st.session_state.page == "settings":
        settings_page()

    else: # Default ke halaman scan jika sudah login
        st.session_state.page = "scan"
        st.rerun()
        
# =====================================================================================
# 6. KONTROLER APLIKASI UTAMA
# =====================================================================================

def welcome_page():
    st.title("Selamat Datang di Nutrisense 🥗")
    st.header("Solusi Cerdas untuk Kesehatan dan Gizi Anda")
    st.markdown(
        """
        Nutrisense membantu Anda memahami hubungan antara makanan yang Anda konsumsi 
        dengan kondisi kesehatan Anda. Cukup dengan satu foto, dapatkan analisis gizi 
        lengkap beserta rekomendasi yang dipersonalisasi untuk Anda.
        """
    )
    col1, col2 = st.columns(2)
    if col1.button("Login", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()
    if col2.button("Register", use_container_width=True):
        st.session_state.page = "register"
        st.rerun()

def auth_page(mode):
    if mode == "login":
        st.subheader("Login ke Akun Anda")
        with st.form("login_form"):
            email = st.text_input("Email"); password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                users = load_users()
                if email in users and users[email]['password'] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    st.session_state.name = users[email].get('name', email)
                    st.session_state.page = "main_app"
                    st.rerun()
                else: st.error("Email atau password salah.")
    
    elif mode == "register":
        st.subheader("Buat Akun Baru")
        with st.form("register_form"):
            name = st.text_input("Nama Lengkap")
            email = st.text_input("Email Baru")
            password = st.text_input("Buat Password", type="password")
            confirm_password = st.text_input("Konfirmasi Password", type="password")
            if st.form_submit_button("Register", type="primary"):
                users = load_users()
                if email in users: st.error("Email sudah terdaftar.")
                elif password != confirm_password: st.error("Password tidak cocok.")
                else:
                    users[email] = {"password": hash_password(password), "name": name, "health_data": None}
                    save_users(users); st.success("Registrasi berhasil! Silakan login.")
                    st.session_state.page = "login"
                    st.rerun()

    if st.button("Kembali ke Halaman Utama"):
        st.session_state.page = "welcome"
        st.rerun()

def initial_health_data_entry():
    st.title("Satu Langkah Lagi!"); st.header("Masukkan Data Kesehatan Awal Anda")
    with st.form("initial_health_form"):
        # Form input data kesehatan...
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            height = h_col1.number_input('Tinggi Badan (cm)', 160)
            weight = h_col1.number_input('Berat Badan (kg)', 60)
            gender = h_col1.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        with h_col2:
            age = h_col2.number_input('Usia', 25)
            bp = h_col2.number_input('Tekanan Darah Sistolik', 120)
            chol = h_col2.number_input('Kolesterol Total', 180)
            glucose = h_col2.number_input('Gula Darah Puasa', 90)
        
        if st.form_submit_button("Simpan dan Lanjutkan", type="primary"):
            health_data = {'Tinggi Badan (cm)': height, 'Berat Badan (kg)': weight, 'Jenis Kelamin': gender, 'Usia': age, 
                           'Tekanan Darah Sistolik (mmHg)': bp, 'Kolesterol Total (mg/dL)': chol, 'Gula Darah Puasa (mg/dL)': glucose}
            users = load_users()
            users[st.session_state['email']]['health_data'] = health_data
            save_users(users)
            st.session_state.page = "main_app"
            st.rerun()


# Inisialisasi session state
if 'page' not in st.session_state:
    st.session_state.page = "welcome"
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Router utama
if not st.session_state.logged_in:
    if st.session_state.page in ["login", "register"]:
        auth_page(st.session_state.page)
    else:
        welcome_page()
else:
    # Cek apakah pengguna perlu mengisi data kesehatan awal
    user_health_data = load_users().get(st.session_state.email, {}).get("health_data")
    if user_health_data is None:
        initial_health_data_entry()
    else:
        main_app()
