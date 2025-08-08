import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from PIL import Image # Ditambahkan untuk memproses gambar

# Konfigurasi Halaman Aplikasi
st.set_page_config(
    page_title="Health Prediction App",
    page_icon="🏥",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT MODEL (DENGAN CACHE) ---

@st.cache_resource
def load_symptoms_model():
    # Konfigurasi kustom untuk optimizer Adam
    class CustomAdam(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            params_to_remove = ['weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 'jit_compile', 'is_legacy_optimizer']
            for param in params_to_remove:
                if param in kwargs:
                    del kwargs[param]
            super().__init__(*args, **kwargs)
    
    custom_objects = {'Custom>Adam': CustomAdam}
    
    # Memuat model prediksi gejala
    model = tf.keras.models.load_model('models/symptoms_predict_model.h5',
                                       custom_objects=custom_objects,
                                       compile=False)
    
    symptoms_idx = {symptom: idx for idx, symptom in enumerate(sorted(symptoms_mapping.values()))}
    
    with open('models/symptoms_labels.json', 'r') as f:
        disease_mapping = json.load(f)
        
    return model, symptoms_idx, disease_mapping

@st.cache_resource
def load_disease_model():
    # Konfigurasi kustom untuk optimizer Adam
    class CustomAdam(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            params_to_remove = ['weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 'jit_compile', 'is_legacy_optimizer']
            for param in params_to_remove:
                if param in kwargs:
                    del kwargs[param]
            super().__init__(*args, **kwargs)
    
    custom_objects = {'CustomAdam': CustomAdam}
    
    return tf.keras.models.load_model('models/disease-prediction-tf-model.h5',
                                      custom_objects=custom_objects,
                                      compile=False)

# --- DATA MAPPING ---

symptoms_mapping = {
    'gatal': 'itching', 'ruam kulit': 'skin_rash', 'benjolan pada kulit': 'nodal_skin_eruptions',
    'jerawat bernanah': 'pus_filled_pimples', 'komedo': 'blackheads', 'kulit mengelupas': 'skin_peeling',
    'kulit seperti berdebu perak': 'silver_like_dusting', 'luka merah di sekitar hidung': 'red_sore_around_nose',
    'keropeng kuning': 'yellow_crust_ooze', 'bersin terus menerus': 'continuous_sneezing',
    'menggigil': 'shivering', 'meriang': 'chills', 'nyeri sendi': 'joint_pain',
    'sakit perut': 'stomach_pain', 'asam lambung': 'acidity', 'sariawan': 'ulcers_on_tongue',
    'otot mengecil': 'muscle_wasting', 'muntah': 'vomiting',
    'rasa terbakar saat buang air kecil': 'burning_micturition', 'bercak saat buang air kecil': 'spotting_urination',
    'kelelahan': 'fatigue', 'kenaikan berat badan': 'weight_gain', 'penurunan berat badan': 'weight_loss',
    'kecemasan': 'anxiety', 'tangan dan kaki dingin': 'cold_hands_and_feets',
    'perubahan suasana hati': 'mood_swings', 'gelisah': 'restlessness', 'lesu': 'lethargy',
    'bercak di tenggorokan': 'patches_in_throat', 'batuk': 'cough', 'sesak napas': 'breathlessness',
    'berkeringat': 'sweating', 'dehidrasi': 'dehydration', 'gangguan pencernaan': 'indigestion',
    'sakit kepala': 'headache', 'kulit kuning': 'yellowish_skin', 'urin gelap': 'dark_urine',
    'mual': 'nausea', 'kehilangan nafsu makan': 'loss_of_appetite',
    'nyeri di belakang mata': 'pain_behind_the_eyes', 'nyeri punggung': 'back_pain', 'sembelit': 'constipation',
    'nyeri perut': 'abdominal_pain', 'diare': 'diarrhoea', 'demam ringan': 'mild_fever',
    'demam tinggi': 'high_fever', 'mata cekung': 'sunken_eyes', 'urin kuning': 'yellow_urine',
    'mata kuning': 'yellowing_of_eyes', 'gagal hati akut': 'acute_liver_failure', 'kelebihan cairan': 'fluid_overload',
    'perut membengkak': 'swelling_of_stomach', 'pembengkakan kelenjar getah bening': 'swelled_lymph_nodes',
    'malaise': 'malaise', 'penglihatan kabur': 'blurred_and_distorted_vision', 'dahak': 'phlegm',
    'iritasi tenggorokan': 'throat_irritation', 'mata merah': 'redness_of_eyes', 'tekanan sinus': 'sinus_pressure',
    'hidung berair': 'runny_nose', 'hidung tersumbat': 'congestion', 'nyeri dada': 'chest_pain',
    'kelemahan anggota tubuh': 'weakness_in_limbs', 'detak jantung cepat': 'fast_heart_rate',
    'nyeri saat buang air besar': 'pain_during_bowel_movements', 'nyeri di daerah anus': 'pain_in_anal_region',
    'tinja berdarah': 'bloody_stool', 'iritasi pada anus': 'irritation_in_anus', 'nyeri leher': 'neck_pain',
    'pusing': 'dizziness', 'kram': 'cramps', 'memar': 'bruising', 'obesitas': 'obesity',
    'kaki bengkak': 'swollen_legs', 'pembuluh darah bengkak': 'swollen_blood_vessels',
    'wajah dan mata bengkak': 'puffy_face_and_eyes', 'kelenjar tiroid membesar': 'enlarged_thyroid',
    'kuku rapuh': 'brittle_nails', 'ekstremitas bengkak': 'swollen_extremeties',
    'rasa lapar berlebihan': 'excessive_hunger', 'bibir kering dan kesemutan': 'drying_and_tingling_lips',
    'bicara pelo': 'slurred_speech', 'nyeri lutut': 'knee_pain', 'nyeri sendi pinggul': 'hip_joint_pain',
    'kelemahan otot': 'muscle_weakness', 'leher kaku': 'stiff_neck', 'sendi bengkak': 'swelling_joints',
    'kekakuan gerakan': 'movement_stiffness', 'gerakan berputar': 'spinning_movements',
    'kehilangan keseimbangan': 'loss_of_balance', 'goyah': 'unsteadiness',
    'kelemahan satu sisi tubuh': 'weakness_of_one_body_side', 'kehilangan penciuman': 'loss_of_smell',
    'ketidaknyamanan kandung kemih': 'bladder_discomfort', 'bau urin tidak sedap': 'foul_smell_of urine',
    'buang air kecil terus menerus': 'continuous_feel_of_urine', 'buang gas': 'passage_of_gases',
    'gatal internal': 'internal_itching', 'wajah toksik': 'toxic_look_(typhos)', 'depresi': 'depression',
    'mudah tersinggung': 'irritability', 'nyeri otot': 'muscle_pain',
    'perubahan kesadaran': 'altered_sensorium', 'bintik merah di tubuh': 'red_spots_over_body',
    'menstruasi tidak normal': 'abnormal_menstruation', 'bercak perubahan warna': 'dischromic_patches',
    'mata berair': 'watering_from_eyes', 'nafsu makan meningkat': 'increased_appetite',
    'buang air kecil berlebihan': 'polyuria', 'riwayat keluarga': 'family_history',
    'dahak berlendir': 'mucoid_sputum', 'dahak berkarat': 'rusty_sputum',
    'kurang konsentrasi': 'lack_of_concentration', 'gangguan penglihatan': 'visual_disturbances',
    'menerima transfusi darah': 'receiving_blood_transfusion',
    'menerima suntikan tidak steril': 'receiving_unsterile_injections', 'koma': 'coma',
    'pendarahan lambung': 'stomach_bleeding', 'perut membuncit': 'distention_of_abdomen',
    'riwayat konsumsi alkohol': 'history_of_alcohol_consumption', 'dahak berdarah': 'blood_in_sputum',
    'pembuluh darah menonjol di betis': 'prominent_veins_on_calf', 'jantung berdebar': 'palpitations',
    'nyeri saat berjalan': 'painful_walking', 'lepuh': 'blister'
}

# --- FUNGSI UNTUK SETIAP HALAMAN ---

def symptoms_analysis_page():
    st.title("Analisis Gejala")
    
    model, symptoms_idx, disease_mapping = load_symptoms_model()
    
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cari gejala:")
        symptom = st.selectbox("", list(symptoms_mapping.keys()))
        if st.button("Tambah Gejala"):
            if symptom not in st.session_state.selected_symptoms:
                st.session_state.selected_symptoms.append(symptom)
                st.rerun()

    with col2:
        st.subheader("Gejala yang dipilih:")
        for i, symptom in enumerate(st.session_state.selected_symptoms):
            col_left, col_right = st.columns([6,1])
            with col_left:
                st.write(f"• {symptom}")
            with col_right:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.selected_symptoms.remove(symptom)
                    st.rerun()
    
    if st.button("Analisis Penyakit", type="primary"):
        if len(st.session_state.selected_symptoms) > 0:
            input_symptoms = np.zeros(132)
            for symptom in st.session_state.selected_symptoms:
                eng_symptom = symptoms_mapping[symptom]
                if eng_symptom in symptoms_idx:
                    input_symptoms[symptoms_idx[eng_symptom]] = 1

            input_symptoms = np.array([input_symptoms])
            predictions = model.predict(input_symptoms)
            
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            
            st.subheader("Hasil Analisis:")
            for idx in top_3_idx:
                disease = disease_mapping["idx_to_disease"][str(idx)]
                probability = predictions[0][idx] * 100
                st.write(f"• **{disease}**: {probability:.2f}%")
        else:
            st.warning("Silakan pilih minimal satu gejala terlebih dahulu.")

def calculate_derived_features(height, weight, gender, age, blood_pressure, cholesterol, blood_glucose):
    height_m = height / 100
    bmi = weight / (height_m ** 2) if height_m > 0 else 0
    
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
        'bp_category': bp_category, 'bmi_age': bmi_age, 'bp_age': bp_age,
        'bmi_bp': bmi_bp, 'sodium': sodium, 'fat': fat, 'protein': protein,
        'carbs': carbs
    }

def disease_prediction_page():
    st.title('Prediksi Penyakit dari Data Kesehatan')
    st.write('Masukkan data kesehatan Anda untuk mendapatkan prediksi penyakit.')
    
    col1, col2 = st.columns(2)
    
    with col1:
        height = st.number_input('Tinggi Badan (cm)', min_value=50, max_value=300, value=165, step=1)
        weight = st.number_input('Berat Badan (kg)', min_value=10, max_value=500, value=65, step=1)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        age = st.number_input('Usia', min_value=1, max_value=150, value=25, step=1)
        
    with col2:
        blood_pressure = st.number_input('Tekanan Darah Sistolik (mmHg)', min_value=50, max_value=300, value=120, step=1)
        cholesterol = st.number_input('Kolesterol Total (mg/dL)', min_value=50, max_value=1000, value=170, step=1)
        blood_glucose = st.number_input('Gula Darah Puasa (mg/dL)', min_value=50, max_value=1000, value=80, step=1)
    
    if st.button('Analisis Kesehatan', type="primary"):
        gender_binary = 1 if gender == 'Laki-laki' else 0
        
        derived = calculate_derived_features(height, weight, gender_binary, age, blood_pressure, cholesterol, blood_glucose)
        
        features = np.array([[
            height, weight, gender_binary, age, blood_pressure, cholesterol, blood_glucose,
            derived['bmi'], derived['bmi_category'], derived['age_category'], derived['bp_category'],
            derived['bmi_age'], derived['bp_age'], derived['bmi_bp'], derived['sodium']
        ]])
        
        model = load_disease_model()
        predictions = model.predict(features)
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader('Hasil Analisis BMI')
            st.write(f"**BMI**: {derived['bmi']:.1f}")
            bmi_status = ('Kurus' if derived['bmi'] < 18.5 else 'Normal' if derived['bmi'] < 25 else 'Gemuk' if derived['bmi'] < 30 else 'Obesitas')
            st.write(f"**Status BMI**: {bmi_status}")
            
            st.subheader('Kebutuhan Nutrisi Harian (Estimasi)')
            st.write(f"Sodium: {derived['sodium']:.1f} mg, Lemak: {derived['fat']:.1f} g, Protein: {derived['protein']:.1f} g, Karbohidrat: {derived['carbs']:.1f} g")
        
        with res_col2:
            st.subheader('Risiko Penyakit')
            diseases = ['Anemia', 'Kolesterol', 'CKD', 'Diabetes', 'Jantung', 'Hipertensi', 'MS', 'NAFLD', 'Obesitas', 'Stroke']
            
            high_risk_diseases = [f"• **{disease}** (Risiko: {prob*100:.1f}%)" for disease, prob in zip(diseases, predictions[0]) if prob > 0.5]
            
            if high_risk_diseases:
                for disease_risk in high_risk_diseases:
                    st.write(disease_risk)
            else:
                st.success("Tidak ada risiko penyakit signifikan yang terdeteksi.")

def food_detection_page():
    st.title('Deteksi Makanan dari Gambar')
    st.write("Unggah gambar makanan untuk dideteksi oleh model TFLite.")

    tflite_model_path = 'models/bestmodel.tflite'

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if not os.path.exists(tflite_model_path):
        st.error(f"Model tidak ditemukan di path: {tflite_model_path}")
        st.info("Pastikan Anda telah menempatkan 'bestmodel.tflite' di dalam folder 'models'.")
        return

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)
        
        if st.button('Mulai Deteksi Makanan', type="primary"):
            with st.spinner('Model sedang memproses gambar...'):
                interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                input_shape = input_details[0]['shape']
                img_resized = image.resize((input_shape[1], input_shape[2]))
                
                input_data = np.expand_dims(img_resized, axis=0)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                with col2:
                    st.subheader("Hasil Deteksi")
                    st.write("Output mentah dari model:")
                    st.write(output_data)
                    st.info("Anda perlu memproses output ini lebih lanjut untuk menampilkan nama makanan dan kotak pembatas pada gambar.")

# --- FUNGSI UTAMA UNTUK MENJALANKAN APLIKASI ---

def main():
    st.sidebar.title('Navigasi 🧭')
    page = st.sidebar.radio('Pilih Halaman', ['Prediksi Penyakit', 'Analisis Gejala', 'Deteksi Makanan'])
    
    if page == 'Prediksi Penyakit':
        disease_prediction_page()
    elif page == 'Analisis Gejala':
        symptoms_analysis_page()
    else:
        food_detection_page()

if __name__ == '__main__':
    main()
