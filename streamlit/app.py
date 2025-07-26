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
    layout="centered"
)

# Lokasi file
USER_DATA_FILE = "data/users.json"
SCAN_HISTORY_DIR = "data/scan_history_images/"

# Inisialisasi folder & file
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(SCAN_HISTORY_DIR, exist_ok=True)
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump({}, f)

# Database Gizi Sederhana
NUTRITION_DB = {
    'apple': {'Kalori': 52, 'Karbohidrat (g)': 14, 'Protein (g)': 0.3, 'Lemak (g)': 0.2},
    'banana': {'Kalori': 89, 'Karbohidrat (g)': 23, 'Protein (g)': 1.1, 'Lemak (g)': 0.3},
    'pizza': {'Kalori': 266, 'Karbohidrat (g)': 33, 'Protein (g)': 11, 'Lemak (g)': 10},
}
FOOD_CLASSES = {0: 'background', 1: 'apple', 2: 'banana', 3: 'pizza'}

# =====================================================================================
# FUNGSI MANAJEMEN PENGGUNA
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
    if 'scan_history' not in user_data: user_data['scan_history'] = []
    scan_record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": image_path, "detected_foods": detected_foods, "nutrition_info": nutrition_info
    }
    user_data['scan_history'].append(scan_record)
    users[email] = user_data
    save_users(users)

# =====================================================================================
# MAPPING GEJALA
# =====================================================================================
symptoms_mapping = {
    'gatal': 'itching', 'ruam kulit': 'skin_rash', 'benjolan pada kulit': 'nodal_skin_eruptions',
    'jerawat bernanah': 'pus_filled_pimples', 'komedo': 'blackheads', 'kulit mengelupas': 'skin_peeling',
    'kulit seperti berdebu perak': 'silver_like_dusting', 'luka merah di sekitar hidung': 'red_sore_around_nose',
    'keropeng kuning': 'yellow_crust_ooze','bersin terus menerus': 'continuous_sneezing', 'menggigil': 'shivering',
    'meriang': 'chills', 'nyeri sendi': 'joint_pain', 'sakit perut': 'stomach_pain',
    'asam lambung': 'acidity', 'sariawan': 'ulcers_on_tongue', 'otot mengecil': 'muscle_wasting',
    'muntah': 'vomiting', 'rasa terbakar saat buang air kecil': 'burning_micturition', 'bercak saat buang air kecil': 'spotting_urination',
    'kelelahan': 'fatigue','kenaikan berat badan': 'weight_gain', 'penurunan berat badan': 'weight_loss',
    'kecemasan': 'anxiety', 'tangan dan kaki dingin': 'cold_hands_and_feets', 'perubahan suasana hati': 'mood_swings',
    'gelisah': 'restlessness', 'lesu': 'lethargy','bercak di tenggorokan': 'patches_in_throat', 'batuk': 'cough',
    'sesak napas': 'breathlessness', 'berkeringat': 'sweating', 'dehidrasi': 'dehydration',
    'gangguan pencernaan': 'indigestion', 'sakit kepala': 'headache','kulit kuning': 'yellowish_skin',
    'urin gelap': 'dark_urine', 'mual': 'nausea', 'kehilangan nafsu makan': 'loss_of_appetite',
    'nyeri di belakang mata': 'pain_behind_the_eyes', 'nyeri punggung': 'back_pain', 'sembelit': 'constipation',
    'nyeri perut': 'abdominal_pain', 'diare': 'diarrhoea','demam ringan': 'mild_fever',
    'demam tinggi': 'high_fever', 'mata cekung': 'sunken_eyes', 'urin kuning': 'yellow_urine',
    'mata kuning': 'yellowing_of_eyes', 'gagal hati akut': 'acute_liver_failure','kelebihan cairan': 'fluid_overload',
    'perut membengkak': 'swelling_of_stomach', 'pembengkakan kelenjar getah bening': 'swelled_lymph_nodes',
    'malaise': 'malaise', 'penglihatan kabur': 'blurred_and_distorted_vision', 'dahak': 'phlegm',
    'iritasi tenggorokan': 'throat_irritation', 'mata merah': 'redness_of_eyes', 'tekanan sinus': 'sinus_pressure',
    'hidung berair': 'runny_nose', 'hidung tersumbat': 'congestion', 'nyeri dada': 'chest_pain',
    'kelemahan anggota tubuh': 'weakness_in_limbs', 'detak jantung cepat': 'fast_heart_rate','nyeri saat buang air besar': 'pain_during_bowel_movements',
    'nyeri di daerah anus': 'pain_in_anal_region', 'tinja berdarah': 'bloody_stool', 'iritasi pada anus': 'irritation_in_anus',
    'nyeri leher': 'neck_pain', 'pusing': 'dizziness', 'kram': 'cramps', 'memar': 'bruising', 'obesitas': 'obesity',
    'kaki bengkak': 'swollen_legs', 'pembuluh darah bengkak': 'swollen_blood_vessels', 'wajah dan mata bengkak': 'puffy_face_and_eyes',
    'kelenjar tiroid membesar': 'enlarged_thyroid', 'kuku rapuh': 'brittle_nails', 'ekstremitas bengkak': 'swollen_extremeties',
    'rasa lapar berlebihan': 'excessive_hunger', 'bibir kering dan kesemutan': 'drying_and_tingling_lips',
    'bicara pelo': 'slurred_speech', 'nyeri lutut': 'knee_pain', 'nyeri sendi pinggul': 'hip_joint_pain',
    'kelemahan otot': 'muscle_weakness', 'leher kaku': 'stiff_neck', 'sendi bengkak': 'swelling_joints',
    'kekakuan gerakan': 'movement_stiffness', 'gerakan berputar': 'spinning_movements', 'kehilangan keseimbangan': 'loss_of_balance',
    'goyah': 'unsteadiness', 'kelemahan satu sisi tubuh': 'weakness_of_one_body_side', 'kehilangan penciuman': 'loss_of_smell',
    'ketidaknyamanan kandung kemih': 'bladder_discomfort', 'bau urin tidak sedap': 'foul_smell_of urine', 'buang air kecil terus menerus': 'continuous_feel_of_urine',
    'buang gas': 'passage_of_gases', 'gatal internal': 'internal_itching', 'wajah toksik': 'toxic_look_(typhos)', 'depresi': 'depression',
    'mudah tersinggung': 'irritability', 'nyeri otot': 'muscle_pain', 'perubahan kesadaran': 'altered_sensorium',
    'bintik merah di tubuh': 'red_spots_over_body', 'nyeri perut': 'belly_pain', 'menstruasi tidak normal': 'abnormal_menstruation',
    'bercak perubahan warna': 'dischromic_patches', 'mata berair': 'watering_from_eyes', 'nafsu makan meningkat': 'increased_appetite',
    'buang air kecil berlebihan': 'polyuria', 'riwayat keluarga': 'family_history', 'dahak berlendir': 'mucoid_sputum',
    'dahak berkarat': 'rusty_sputum', 'kurang konsentrasi': 'lack_of_concentration', 'gangguan penglihatan': 'visual_disturbances',
    'menerima transfusi darah': 'receiving_blood_transfusion', 'menerima suntikan tidak steril': 'receiving_unsterile_injections',
    'koma': 'coma', 'pendarahan lambung': 'stomach_bleeding', 'perut membuncit': 'distention_of_abdomen',
    'riwayat konsumsi alkohol': 'history_of_alcohol_consumption', 'dahak berdarah': 'blood_in_sputum',
    'pembuluh darah menonjol di betis': 'prominent_veins_on_calf', 'jantung berdebar': 'palpitations', 'nyeri saat berjalan': 'painful_walking',
    'lepuh': 'blister'
}

# =====================================================================================
# FUNGSI MEMUAT MODEL (VERSI FINAL)
# =====================================================================================
@st.cache_resource
def load_model_for_inference(model_path):
    """Fungsi umum untuk memuat model H5 hanya untuk prediksi."""
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di: {model_path}")
        return None
    try:
        # Cara paling aman: muat tanpa mengkompilasi optimizer.
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model dari {model_path}. Error: {e}")
        return None

@st.cache_resource
def get_symptoms_data():
    """Memuat data pendukung untuk model gejala."""
    labels_path = 'models/symptoms_labels.json'
    if not os.path.exists(labels_path):
        st.error(f"File label tidak ditemukan di: {labels_path}")
        return None, None
    symptoms_idx = {symptom: idx for idx, symptom in enumerate(sorted(symptoms_mapping.values()))}
    with open(labels_path, 'r') as f:
        disease_mapping = json.load(f)
    return symptoms_idx, disease_mapping

# =====================================================================================
# HALAMAN-HALAMAN APLIKASI
# =====================================================================================
def main_dashboard():
    st.set_page_config(layout="wide")
    st.sidebar.success(f"Login sebagai: {st.session_state['email']}")
    page = st.sidebar.radio("Menu Utama", ["🏠 Dashboard", "🩺 Analisis Gejala", "🔬 Prediksi Risiko Penyakit", "📖 Riwayat Scan", "⚙️ Pengaturan"])
    if page == "🏠 Dashboard": display_dashboard()
    elif page == "🩺 Analisis Gejala": symptoms_analysis_page()
    elif page == "🔬 Prediksi Risiko Penyakit": disease_prediction_page()
    elif page == "📖 Riwayat Scan": history_page()
    elif page == "⚙️ Pengaturan": settings_page()
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = None
        st.rerun()

def display_dashboard():
    st.title(f"Selamat Datang, {st.session_state['name']}! 👋")
    st.write("Pusat kendali kesehatan dan nutrisi Anda. Apa yang ingin Anda lakukan hari ini?")
    st.divider()
    st.header("📸 Analisis Gizi Makanan Anda")
    st.write("Gunakan kamera atau unggah gambar untuk mengetahui kandungan gizi makanan Anda secara instan.")
    image_file = st.file_uploader("Unggah gambar makanan...", type=['jpg', 'jpeg', 'png'], key="dashboard_uploader")
    if image_file: handle_food_scan(image_file)

def handle_food_scan(image_file):
    food_model = load_model_for_inference('models/mrcnn_food_detection.h5')
    if food_model is None: 
        st.warning("Fitur deteksi makanan tidak aktif karena model tidak ditemukan.")
        return
    image = Image.open(image_file)
    col1, col2 = st.columns(2)
    with col1: st.image(image, caption="Gambar yang dianalisis", use_column_width=True)
    with col2:
        with st.spinner("Menganalisis makanan..."):
            simulated_detected_foods = ['pizza', 'apple'] 
            st.subheader("✅ Makanan Terdeteksi & Info Gizi")
            all_nutrition_info = {}
            if not simulated_detected_foods: st.warning("Tidak ada makanan yang dapat dikenali.")
            else:
                for food_name in simulated_detected_foods:
                    nutrition_info = NUTRITION_DB.get(food_name.lower())
                    if nutrition_info:
                        st.success(f"**{food_name.capitalize()}**")
                        st.dataframe(pd.DataFrame([nutrition_info]))
                        all_nutrition_info[food_name] = nutrition_info
                    else: st.error(f"Info gizi untuk {food_name.capitalize()} tidak ditemukan.")
            img_path = os.path.join(SCAN_HISTORY_DIR, f"{st.session_state['email'].split('@')[0]}_{int(datetime.datetime.now().timestamp())}.png")
            image.save(img_path)
            save_scan_history(st.session_state['email'], img_path, simulated_detected_foods, all_nutrition_info)
            st.success("Hasil scan berhasil disimpan ke riwayat Anda!")

def symptoms_analysis_page():
    st.title("🩺 Analisis Penyakit Berdasarkan Gejala")
    model = load_model_for_inference('models/symptoms_predict_model.h5')
    symptoms_idx, disease_mapping = get_symptoms_data()
    
    if model is None or symptoms_idx is None: return

    if 'selected_symptoms' not in st.session_state: st.session_state.selected_symptoms = []
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Pilih Gejala Anda:")
        st.session_state.selected_symptoms = st.multiselect("Bisa pilih lebih dari satu", list(symptoms_mapping.keys()), default=st.session_state.selected_symptoms)
    with col2:
        st.subheader("Gejala yang Dipilih:")
        if not st.session_state.selected_symptoms: st.warning("Belum ada gejala dipilih.")
        else:
            for symptom in st.session_state.selected_symptoms: st.info(f"• {symptom}")
    if st.button("Analisis Penyakit Sekarang", type="primary"):
        if st.session_state.selected_symptoms:
            input_symptoms = np.zeros(132)
            for symptom in st.session_state.selected_symptoms:
                english_symptom = symptoms_mapping.get(symptom)
                if english_symptom in symptoms_idx: input_symptoms[symptoms_idx[english_symptom]] = 1
            predictions = model.predict(np.array([input_symptoms]))
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            st.subheader("Hasil Analisis:")
            for i, idx in enumerate(top_3_indices):
                disease_name = disease_mapping["idx_to_disease"][str(idx)]
                probability = predictions[0][idx] * 100
                st.success(f"**{i+1}. {disease_name}** (Keyakinan: {probability:.2f}%)")
        else: st.error("Silakan pilih minimal satu gejala.")

def disease_prediction_page():
    st.title("🔬 Prediksi Risiko Penyakit Kronis")
    st.write("Prediksi dibuat berdasarkan data kesehatan di profil Anda.")
    user_data = load_users()[st.session_state['email']]
    health_data = user_data.get('health_data')
    if not health_data:
        st.warning("Lengkapi data kesehatan di halaman Pengaturan.")
        return
    st.info("Data kesehatan yang digunakan:"); st.json(health_data)
    if st.button("Jalankan Prediksi Risiko", type="primary"):
        model = load_model_for_inference('models/disease-prediction-tf-model.h5')
        if model is None: return
        height = health_data['Tinggi Badan (cm)']; weight = health_data['Berat Badan (kg)']
        gender_binary = 1 if health_data['Jenis Kelamin'] == 'Laki-laki' else 0
        age = health_data['Usia']; bp = health_data['Tekanan Darah Sistolik (mmHg)']
        chol = health_data['Kolesterol Total (mg/dL)']; glucose = health_data['Gula Darah Puasa (mg/dL)']
        bmi = weight / ((height / 100) ** 2)
        bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
        age_category = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
        bp_category = 0 if bp < 120 else 1 if bp < 140 else 2 if bp < 160 else 3
        features = np.array([[height, weight, gender_binary, age, bp, chol, glucose, bmi, bmi_category, age_category, bp_category, bmi * age, bp * age, bmi * bp, weight * 20]])
        predictions = model.predict(features)
        st.subheader("Potensi Risiko Penyakit:")
        diseases = ['Anemia', 'Kolesterol Tinggi', 'Gagal Ginjal Kronis', 'Diabetes', 'Penyakit Jantung', 'Hipertensi', 'Sindrom Metabolik', 'Perlemakan Hati', 'Obesitas', 'Stroke']
        high_risk = [f"• **{diseases[i]}** ({predictions[0][i]*100:.1f}%)" for i, prob in enumerate(predictions[0]) if prob > 0.5]
        if high_risk:
            for risk in high_risk: st.error(risk)
        else: st.success("✅ Tidak ada risiko penyakit kronis yang signifikan terdeteksi.")

def history_page():
    st.title("📖 Riwayat Scan Makanan")
    history = load_users()[st.session_state['email']].get('scan_history', [])
    if not history:
        st.info("Anda belum memiliki riwayat scan.")
        return
    for record in reversed(history):
        with st.expander(f"**Scan pada:** {record['timestamp']}"):
            col1, col2 = st.columns([1,2])
            with col1:
                if os.path.exists(record['image_path']): st.image(record['image_path'])
            with col2:
                st.write("**Makanan:**", ", ".join(record['detected_foods']))
                for food, nutrition in record['nutrition_info'].items():
                    st.write(f"**Gizi {food.capitalize()}:**"); st.json(nutrition)

def settings_page():
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
                    users[email] = {"password": hash_password(password), "name": name, "health_data": None, "scan_history": []}
                    save_users(users); st.success("Registrasi berhasil! Silakan login.")

# =====================================================================================
# MAIN APP ROUTER
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
