import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import gdown
from PIL import Image

# =====================================================================================
# Konfigurasi Halaman dan Judul
# =====================================================================================
st.set_page_config(
    page_title="Tilik Nutrisi",
    page_icon="assets/welcome_image.png",
    layout="centered",
    initial_sidebar_state="auto"
)

# =====================================================================================
# Inisialisasi Session State
# =====================================================================================
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

# =====================================================================================
# Kamus Gejala (Sesuai dengan kode Anda)
# =====================================================================================
symptoms_mapping = {
    'gatal': 'itching', 'ruam kulit': 'skin_rash', 'benjolan pada kulit': 'nodal_skin_eruptions',
    'bersin terus menerus': 'continuous_sneezing', 'menggigil': 'shivering', 'meriang': 'chills',
    'nyeri sendi': 'joint_pain', 'sakit perut': 'stomach_pain', 'asam lambung': 'acidity',
    'sariawan': 'ulcers_on_tongue', 'otot mengecil': 'muscle_wasting', 'muntah': 'vomiting',
    'rasa terbakar saat buang air kecil': 'burning_micturition', 'bercak saat buang air kecil': 'spotting_urination',
    'kelelahan': 'fatigue', 'kenaikan berat badan': 'weight_gain', 'kecemasan': 'anxiety',
    'tangan dan kaki dingin': 'cold_hands_and_feets', 'perubahan suasana hati': 'mood_swings',
    'penurunan berat badan': 'weight_loss', 'gelisah': 'restlessness', 'lesu': 'lethargy',
    'bercak di tenggorokan': 'patches_in_throat', 'kadar gula tidak teratur': 'irregular_sugar_level',
    'batuk': 'cough', 'demam tinggi': 'high_fever', 'mata cekung': 'sunken_eyes',
    'sesak napas': 'breathlessness', 'berkeringat': 'sweating', 'dehidrasi': 'dehydration',
    'gangguan pencernaan': 'indigestion', 'sakit kepala': 'headache', 'kulit kuning': 'yellowish_skin',
    'urin gelap': 'dark_urine', 'mual': 'nausea', 'kehilangan nafsu makan': 'loss_of_appetite',
    'nyeri di belakang mata': 'pain_behind_the_eyes', 'nyeri punggung': 'back_pain', 'sembelit': 'constipation',
    'nyeri perut': 'abdominal_pain', 'diare': 'diarrhoea', 'demam ringan': 'mild_fever',
    'urin kuning': 'yellow_urine', 'mata kuning': 'yellowing_of_eyes', 'gagal hati akut': 'acute_liver_failure',
    'kelebihan cairan': 'fluid_overload', 'perut membengkak': 'swelling_of_stomach',
    'pembengkakan kelenjar getah bening': 'swelled_lymph_nodes', 'malaise': 'malaise',
    'penglihatan kabur': 'blurred_and_distorted_vision', 'dahak': 'phlegm', 'iritasi tenggorokan': 'throat_irritation',
    'mata merah': 'redness_of_eyes', 'tekanan sinus': 'sinus_pressure', 'hidung berair': 'runny_nose',
    'hidung tersumbat': 'congestion', 'nyeri dada': 'chest_pain', 'kelemahan anggota tubuh': 'weakness_in_limbs',
    'detak jantung cepat': 'fast_heart_rate', 'nyeri saat buang air besar': 'pain_during_bowel_movements',
    'nyeri di daerah anus': 'pain_in_anal_region', 'tinja berdarah': 'bloody_stool',
    'iritasi pada anus': 'irritation_in_anus', 'nyeri leher': 'neck_pain', 'pusing': 'dizziness',
    'kram': 'cramps', 'memar': 'bruising', 'obesitas': 'obesity', 'kaki bengkak': 'swollen_legs',
    'pembuluh darah bengkak': 'swollen_blood_vessels', 'wajah dan mata bengkak': 'puffy_face_and_eyes',
    'kelenjar tiroid membesar': 'enlarged_thyroid', 'kuku rapuh': 'brittle_nails',
    'ekstremitas bengkak': 'swollen_extremeties', 'rasa lapar berlebihan': 'excessive_hunger',
    'kontak di luar nikah': 'extra_marital_contacts', 'bibir kering dan kesemutan': 'drying_and_tingling_lips',
    'bicara pelo': 'slurred_speech', 'nyeri lutut': 'knee_pain', 'nyeri sendi pinggul': 'hip_joint_pain',
    'kelemahan otot': 'muscle_weakness', 'leher kaku': 'stiff_neck', 'sendi bengkak': 'swelling_joints',
    'kekakuan gerakan': 'movement_stiffness', 'gerakan berputar': 'spinning_movements',
    'kehilangan keseimbangan': 'loss_of_balance', 'goyah': 'unsteadiness',
    'kelemahan satu sisi tubuh': 'weakness_of_one_body_side', 'kehilangan penciuman': 'loss_of_smell',
    'ketidaknyamanan kandung kemih': 'bladder_discomfort', 'bau urin tidak sedap': 'foul_smell_of urine',
    'buang air kecil terus menerus': 'continuous_feel_of_urine', 'buang gas': 'passage_of_gases',
    'gatal internal': 'internal_itching', 'wajah toksik': 'toxic_look_(typhos)', 'depresi': 'depression',
    'mudah tersinggung': 'irritability', 'nyeri otot': 'muscle_pain', 'perubahan kesadaran': 'altered_sensorium',
    'bintik merah di tubuh': 'red_spots_over_body', 'nyeri perut': 'belly_pain',
    'menstruasi tidak normal': 'abnormal_menstruation', 'bercak perubahan warna': 'dischromic_patches',
    'mata berair': 'watering_from_eyes', 'nafsu makan meningkat': 'increased_appetite',
    'buang air kecil berlebihan': 'polyuria', 'riwayat keluarga': 'family_history', 'dahak berlendir': 'mucoid_sputum',
    'dahak berkarat': 'rusty_sputum', 'kurang konsentrasi': 'lack_of_concentration',
    'gangguan penglihatan': 'visual_disturbances', 'menerima transfusi darah': 'receiving_blood_transfusion',
    'menerima suntikan tidak steril': 'receiving_unsterile_injections', 'koma': 'coma',
    'pendarahan lambung': 'stomach_bleeding', 'perut membuncit': 'distention_of_abdomen',
    'riwayat konsumsi alkohol': 'history_of_alcohol_consumption', 'dahak berdarah': 'blood_in_sputum',
    'pembuluh darah menonjol di betis': 'prominent_veins_on_calf', 'jantung berdebar': 'palpitations',
    'nyeri saat berjalan': 'painful_walking', 'jerawat bernanah': 'pus_filled_pimples', 'komedo': 'blackheads',
    'kulit mengelupas': 'scurring', 'kulit seperti berdebu perak': 'silver_like_dusting',
    'luka merah di sekitar hidung': 'red_sore_around_nose', 'keropeng kuning': 'yellow_crust_ooze',
    'lepuh': 'blister'
}


# =====================================================================================
# Fungsi untuk Memuat Model
# =====================================================================================
@st.cache_resource
def load_all_models():
    """Memuat semua model dan data pendukung."""
    models = {'disease': None, 'symptoms': None, 'classification': None}
    data = {'symptoms_idx': None, 'disease_mapping': None}
    
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- Konfigurasi Optimizer Kustom ---
    class CustomAdam(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            for param in ['weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 'jit_compile', 'is_legacy_optimizer']:
                if param in kwargs:
                    del kwargs[param]
            super().__init__(*args, **kwargs)
    
    custom_objects = {'CustomAdam': CustomAdam, 'Custom>Adam': CustomAdam}

    # --- Model Prediksi Penyakit & Gejala ---
    try:
        if os.path.exists('models/disease-prediction-model.hdf5'):
            models['disease'] = tf.keras.models.load_model('models/disease-prediction-model.hdf5', custom_objects=custom_objects, compile=False)
        
        if os.path.exists('models/symptoms_predict_model.h5'):
            models['symptoms'] = tf.keras.models.load_model('models/symptoms_predict_model.h5', custom_objects=custom_objects, compile=False)
        
        # Muat file JSON untuk model gejala
        if os.path.exists('models/symptoms_labels.json'):
            with open('models/symptoms_labels.json', 'r') as f:
                data['disease_mapping'] = json.load(f)
            # Buat pemetaan indeks dari kamus gejala
            data['symptoms_idx'] = {symptom: idx for idx, symptom in enumerate(sorted(symptoms_mapping.values()))}

    except Exception as e:
        st.error(f"Gagal memuat model penyakit/gejala: {e}")
            
    # --- Model Klasifikasi Makanan ---
    tflite_model_path = 'models/bestmodel.tflite'
    gdrive_file_id = '1fRrv1FCg8hd2uLyBUbhB0DpI6teixUob'
    try:
        if not os.path.exists(tflite_model_path):
            with st.spinner("Mengunduh model klasifikasi makanan..."):
                gdown.download(id=gdrive_file_id, output=tflite_model_path, quiet=False)
        
        models['classification'] = tf.lite.Interpreter(model_path=tflite_model_path)
        models['classification'].allocate_tensors()
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi makanan: {e}")

    return models, data

# Memuat semua aset saat aplikasi dimulai
models, data = load_all_models()
disease_model = models['disease']
symptom_model = models['symptoms']
classification_model = models['classification']
symptoms_idx = data['symptoms_idx']
disease_mapping = data['disease_mapping']


# =====================================================================================
# Halaman Aplikasi
# =====================================================================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Selamat Datang", "Prediksi Gejala", "Prediksi Penyakit", "Analisis Makanan"])

# --- Halaman 1: Selamat Datang ---
if page == "Selamat Datang":
    st.title("Selamat Datang di Tilik Nutrisi 🥗")
    st.markdown("Solusi cerdas untuk memantau kesehatan dan nutrisi Anda.")
    # ... (Sisa kode halaman selamat datang bisa ditambahkan di sini)

# --- Halaman 2: Prediksi Gejala ---
elif page == "Prediksi Gejala":
    st.header("🩺 Prediksi Penyakit Berdasarkan Gejala")
    if symptom_model and symptoms_idx and disease_mapping:
        selected_symptoms = st.multiselect("Pilih gejala yang Anda rasakan:", list(symptoms_mapping.keys()))
        
        if st.button("Analisis Gejala"):
            if selected_symptoms:
                with st.spinner("Menganalisis gejala..."):
                    input_symptoms = np.zeros(len(symptoms_mapping)) # Ukuran harus sesuai dengan jumlah total gejala
                    for symptom in selected_symptoms:
                        eng_symptom = symptoms_mapping[symptom]
                        if eng_symptom in symptoms_idx:
                            input_symptoms[symptoms_idx[eng_symptom]] = 1
                    
                    input_symptoms = np.array([input_symptoms], dtype=np.float32)
                    predictions = symptom_model.predict(input_symptoms)
                    
                    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                    
                    st.subheader("Hasil Analisis:")
                    for idx in top_3_idx:
                        disease = disease_mapping["idx_to_disease"][str(idx)]
                        probability = predictions[0][idx] * 100
                        st.write(f"• **{disease}**: {probability:.2f}%")
            else:
                st.warning("Silakan pilih minimal satu gejala.")
    else:
        st.error("Model atau data pendukung untuk analisis gejala tidak berhasil dimuat.")

# --- Halaman 3: Prediksi Penyakit ---
elif page == "Prediksi Penyakit":
    st.header("🔬 Prediksi Risiko Penyakit dari Data Kesehatan")
    # ... (Kode untuk halaman ini bisa ditambahkan, mirip dengan versi sebelumnya)
    st.info("Fitur ini sedang dalam pengembangan.")


# --- Halaman 4: Analisis Makanan ---
elif page == "Analisis Makanan":
    st.header("📸 Analisis Gizi Makanan")
    # ... (Kode untuk halaman ini bisa ditambahkan, mirip dengan versi sebelumnya)
    st.info("Fitur ini sedang dalam pengembangan.")

