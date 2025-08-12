import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import gdown
from PIL import Image

# ========== CONFIG ==========
MRCNN_MODEL_PATH = "models/mrcnn_food_detection_final.h5"
MRCNN_GDRIVE_URL = "https://drive.google.com/uc?id=1sVzG18PNoUrwxctxIMIoQL79bCWjhnSY"

# Symptoms mapping
symptoms_mapping = {
    'gatal': 'itching',
    'ruam kulit': 'skin_rash', 
    'benjolan pada kulit': 'nodal_skin_eruptions',
    'jerawat bernanah': 'pus_filled_pimples',
    'komedo': 'blackheads',
    'kulit mengelupas': 'skin_peeling',
    'kulit seperti berdebu perak': 'silver_like_dusting',
    'luka merah di sekitar hidung': 'red_sore_around_nose',
    'keropeng kuning': 'yellow_crust_ooze',#
    'bersin terus menerus': 'continuous_sneezing',
    'menggigil': 'shivering',
    'meriang': 'chills',
    'nyeri sendi': 'joint_pain',
    'sakit perut': 'stomach_pain',
    'asam lambung': 'acidity',
    'sariawan': 'ulcers_on_tongue',
    'otot mengecil': 'muscle_wasting',
    'muntah': 'vomiting',
    'rasa terbakar saat buang air kecil': 'burning_micturition',
    'bercak saat buang air kecil': 'spotting_urination',
    'kelelahan': 'fatigue',#
    'kenaikan berat badan': 'weight_gain',
    'penurunan berat badan': 'weight_loss',
    'kecemasan': 'anxiety',
    'tangan dan kaki dingin': 'cold_hands_and_feets',
    'perubahan suasana hati': 'mood_swings',
    'gelisah': 'restlessness',
    'lesu': 'lethargy',#
    'bercak di tenggorokan': 'patches_in_throat',
    'batuk': 'cough',
    'sesak napas': 'breathlessness',
    'berkeringat': 'sweating',
    'dehidrasi': 'dehydration',
    'gangguan pencernaan': 'indigestion',
    'sakit kepala': 'headache',#
    'kulit kuning': 'yellowish_skin',
    'urin gelap': 'dark_urine',
    'mual': 'nausea',
    'kehilangan nafsu makan': 'loss_of_appetite',
    'nyeri di belakang mata': 'pain_behind_the_eyes',
    'nyeri punggung': 'back_pain',
    'sembelit': 'constipation',
    'nyeri perut': 'abdominal_pain',
    'diare': 'diarrhoea',#
    'demam ringan': 'mild_fever',
    'demam tinggi': 'high_fever',
    'mata cekung': 'sunken_eyes',
    'urin kuning': 'yellow_urine',
    'mata kuning': 'yellowing_of_eyes',
    'gagal hati akut': 'acute_liver_failure',#
    'kelebihan cairan': 'fluid_overload',
    'perut membengkak': 'swelling_of_stomach',
    'pembengkakan kelenjar getah bening': 'swelled_lymph_nodes',
    'malaise': 'malaise',
    'penglihatan kabur': 'blurred_and_distorted_vision',
    'dahak': 'phlegm',
    'iritasi tenggorokan': 'throat_irritation',
    'mata merah': 'redness_of_eyes',
    'tekanan sinus': 'sinus_pressure',
    'hidung berair': 'runny_nose',
    'hidung tersumbat': 'congestion',
    'nyeri dada': 'chest_pain',
    'kelemahan anggota tubuh': 'weakness_in_limbs',
    'detak jantung cepat': 'fast_heart_rate',#
    'nyeri saat buang air besar': 'pain_during_bowel_movements',
    'nyeri di daerah anus': 'pain_in_anal_region',
    'tinja berdarah': 'bloody_stool',
    'iritasi pada anus': 'irritation_in_anus',
    'nyeri leher': 'neck_pain',
    'pusing': 'dizziness',
    'kram': 'cramps',
    'memar': 'bruising',
    'obesitas': 'obesity',
    'kaki bengkak': 'swollen_legs',
    'pembuluh darah bengkak': 'swollen_blood_vessels',
    'wajah dan mata bengkak': 'puffy_face_and_eyes',
    'kelenjar tiroid membesar': 'enlarged_thyroid',
    'kuku rapuh': 'brittle_nails',
    'ekstremitas bengkak': 'swollen_extremeties',
    'rasa lapar berlebihan': 'excessive_hunger',
    'bibir kering dan kesemutan': 'drying_and_tingling_lips',
    'bicara pelo': 'slurred_speech',
    'nyeri lutut': 'knee_pain',
    'nyeri sendi pinggul': 'hip_joint_pain',
    'kelemahan otot': 'muscle_weakness',
    'leher kaku': 'stiff_neck',
    'sendi bengkak': 'swelling_joints',
    'kekakuan gerakan': 'movement_stiffness',
    'gerakan berputar': 'spinning_movements',
    'kehilangan keseimbangan': 'loss_of_balance',
    'goyah': 'unsteadiness',
    'kelemahan satu sisi tubuh': 'weakness_of_one_body_side',
    'kehilangan penciuman': 'loss_of_smell',
    'ketidaknyamanan kandung kemih': 'bladder_discomfort',
    'bau urin tidak sedap': 'foul_smell_of urine',
    'buang air kecil terus menerus': 'continuous_feel_of_urine',
    'buang gas': 'passage_of_gases',
    'gatal internal': 'internal_itching',
    'wajah toksik': 'toxic_look_(typhos)',
    'depresi': 'depression',
    'mudah tersinggung': 'irritability',
    'nyeri otot': 'muscle_pain',
    'perubahan kesadaran': 'altered_sensorium',
    'bintik merah di tubuh': 'red_spots_over_body',
    'nyeri perut': 'belly_pain',
    'menstruasi tidak normal': 'abnormal_menstruation',
    'bercak perubahan warna': 'dischromic_patches',
    'mata berair': 'watering_from_eyes',
    'nafsu makan meningkat': 'increased_appetite',
    'buang air kecil berlebihan': 'polyuria',
    'riwayat keluarga': 'family_history',
    'dahak berlendir': 'mucoid_sputum',
    'dahak berkarat': 'rusty_sputum',
    'kurang konsentrasi': 'lack_of_concentration',
    'gangguan penglihatan': 'visual_disturbances',
    'menerima transfusi darah': 'receiving_blood_transfusion',
    'menerima suntikan tidak steril': 'receiving_unsterile_injections',
    'koma': 'coma',
    'pendarahan lambung': 'stomach_bleeding',
    'perut membuncit': 'distention_of_abdomen',
    'riwayat konsumsi alkohol': 'history_of_alcohol_consumption',
    'dahak berdarah': 'blood_in_sputum',
    'pembuluh darah menonjol di betis': 'prominent_veins_on_calf',
    'jantung berdebar': 'palpitations',
    'nyeri saat berjalan': 'painful_walking',
    'lepuh': 'blister'
}


# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Tilik Nutrisi", page_icon="🥗", layout="wide")

# ========== MODEL LOADERS ==========
@st.cache_resource
def load_disease_model():
    class CustomAdam(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            for param in ['weight_decay','use_ema','ema_momentum','ema_overwrite_frequency','jit_compile','is_legacy_optimizer']:
                if param in kwargs:
                    del kwargs[param]
            super().__init__(*args, **kwargs)
    return tf.keras.models.load_model(
        "models/disease-prediction-tf-model.h5",
        custom_objects={"CustomAdam": CustomAdam},
        compile=False
    )

@st.cache_resource
def load_symptoms_model():
    class CustomAdam(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            for param in ['weight_decay','use_ema','ema_momentum','ema_overwrite_frequency','jit_compile','is_legacy_optimizer']:
                if param in kwargs:
                    del kwargs[param]
            super().__init__(*args, **kwargs)
    with open('models/symptoms_labels.json', 'r') as f:
        disease_mapping = json.load(f)
    symptoms_idx = {v: i for i, v in enumerate(sorted(set(symptoms_mapping.values())))}
    model = tf.keras.models.load_model(
        "models/symptoms_predict_model.h5",
        custom_objects={"Custom>Adam": CustomAdam},
        compile=False
    )
    return model, symptoms_idx, disease_mapping

@st.cache_resource
def load_food_model():
    # Download MRCNN model if not exists
    if not os.path.exists(MRCNN_MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        gdown.download(MRCNN_GDRIVE_URL, MRCNN_MODEL_PATH, quiet=False)
    from mrcnn.config import Config
    from mrcnn import model as modellib
    class InferenceConfig(Config):
        NAME = "food_cfg"
        NUM_CLASSES = 1 + 80
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=".")
    model.load_weights(MRCNN_MODEL_PATH, by_name=True)
    return model

# ========== HELPER ==========
def calculate_derived_features(height, weight, gender, age, blood_pressure, cholesterol, blood_glucose):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
    age_category = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
    bp_category = 0 if blood_pressure < 120 else 1 if blood_pressure < 140 else 2 if blood_pressure < 160 else 3
    bmi_age = bmi * age
    bp_age = blood_pressure * age
    bmi_bp = bmi * blood_pressure
    sodium = weight * 20
    fat = weight * (0.15 if gender == 1 else 0.25)
    protein = weight * 0.9
    carbs = weight * 3
    return {
        'bmi': bmi, 'bmi_category': bmi_category, 'age_category': age_category,
        'bp_category': bp_category, 'bmi_age': bmi_age, 'bp_age': bp_age, 'bmi_bp': bmi_bp,
        'sodium': sodium, 'fat': fat, 'protein': protein, 'carbs': carbs
    }

# ========== PAGES ==========
def page_welcome():
    st.title("Selamat Datang di Tilik Nutrisi 🥗")
    st.write("Smart Solution for Disease Detection and Food/Nutrition Recommendation.")

def page_register():
    st.title("Daftar Akun")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Daftar"):
        st.success(f"Akun {username} berhasil dibuat! Silakan login.")

def page_login():
    st.title("Masuk")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state["logged_in"] = True
        st.success("Login berhasil!")

def page_disease_prediction():
    st.title("Prediksi Penyakit")
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input('Tinggi Badan (cm)', value=165)
        weight = st.number_input('Berat Badan (kg)', value=65)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        age = st.number_input('Usia', value=25)
    with col2:
        bp = st.number_input('Tekanan Darah Sistolik', value=120)
        chol = st.number_input('Kolesterol Total', value=180)
        glu = st.number_input('Gula Darah Puasa', value=90)
    if st.button("Analisis"):
        model = load_disease_model()
        gender_bin = 1 if gender == 'Laki-laki' else 0
        derived = calculate_derived_features(height, weight, gender_bin, age, bp, chol, glu)
        features = np.array([[height, weight, gender_bin, age, bp, chol, glu,
                              derived['bmi'], derived['bmi_category'],
                              derived['age_category'], derived['bp_category'],
                              derived['bmi_age'], derived['bp_age'],
                              derived['bmi_bp'], derived['sodium']]])
        preds = model.predict(features)
        st.subheader("Hasil Prediksi")
        diseases = ['Anemia','Kolesterol','CKD','Diabetes','Jantung','Hipertensi','MS','NAFLD','Obesitas','Stroke']
        for dis, prob in zip(diseases, preds[0]):
            st.write(f"{dis}: {prob*100:.2f}%")

def page_symptoms_analysis():
    st.title("Analisis Gejala")
    model, symptoms_idx, disease_mapping = load_symptoms_model()
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    symptom = st.selectbox("Pilih gejala", list(symptoms_mapping.keys()))
    if st.button("Tambah Gejala"):
        if symptom not in st.session_state.selected_symptoms:
            st.session_state.selected_symptoms.append(symptom)
    st.write("Gejala terpilih:", st.session_state.selected_symptoms)
    if st.button("Analisis Penyakit"):
        if st.session_state.selected_symptoms:
            input_sym = np.zeros(len(symptoms_idx))
            for s in st.session_state.selected_symptoms:
                eng = symptoms_mapping[s]
                if eng in symptoms_idx:
                    input_sym[symptoms_idx[eng]] = 1
            preds = model.predict(np.array([input_sym]))
            top3 = np.argsort(preds[0])[-3:][::-1]
            for idx in top3:
                disease = disease_mapping["idx_to_disease"][str(idx)]
                st.write(f"{disease}: {preds[0][idx]*100:.2f}%")
        else:
            st.warning("Pilih minimal satu gejala.")

def page_food_detection():
    st.title("Deteksi Makanan")
    model = load_food_model()
    img_file = st.file_uploader("Upload gambar makanan", type=["jpg","jpeg","png"])
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Gambar Input", use_column_width=True)
        image_np = np.array(image)
        results = model.detect([image_np], verbose=0)
        r = results[0]
        st.write("Deteksi selesai. Jumlah objek:", len(r['class_ids']))

# ========== MAIN ==========
menu = st.sidebar.radio("Navigasi", ["Welcome","Login","Register","Disease Prediction","Symptoms Analysis","Food Detection"])
if menu == "Welcome":
    page_welcome()
elif menu == "Login":
    page_login()
elif menu == "Register":
    page_register()
elif menu == "Disease Prediction":
    page_disease_prediction()
elif menu == "Symptoms Analysis":
    page_symptoms_analysis()
elif menu == "Food Detection":
    page_food_detection()
