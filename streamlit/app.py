import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os

# Set page config
st.set_page_config(
    page_title="Health Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_symptoms_model():
    # Custom optimizer configuration to handle potential loading issues
    class CustomAdam(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            # Remove parameters not recognized by the standard loader
            for param in ['weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 'jit_compile', 'is_legacy_optimizer']:
                if param in kwargs:
                    del kwargs[param]
            super().__init__(*args, **kwargs)
    
    custom_objects = {
        'Custom>Adam': CustomAdam
    }
    
    # Load the symptoms prediction model
    model = tf.keras.models.load_model('models/symptoms_predict_model.h5',
                                       custom_objects=custom_objects,
                                       compile=False)
    
    # Load symptoms index mapping from the main mapping dictionary
    # Note: Ensure 'symptoms_mapping' is available globally or passed as an argument
    symptoms_idx = {symptom: idx for idx, symptom in enumerate(sorted(symptoms_mapping.values()))}
    
    # Load disease labels
    with open('models/symptoms_labels.json', 'r') as f:
        disease_mapping = json.load(f)
        
    return model, symptoms_idx, disease_mapping

@st.cache_resource
def load_disease_model():
    # Custom optimizer configuration
    class CustomAdam(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            for param in ['weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 'jit_compile', 'is_legacy_optimizer']:
                if param in kwargs:
                    del kwargs[param]
            super().__init__(*args, **kwargs)
    
    custom_objects = {
        'CustomAdam': CustomAdam
    }
    
    return tf.keras.models.load_model('models/disease-prediction-tf-model.h5',
                                      custom_objects=custom_objects,
                                      compile=False)

# Symptoms mapping dictionary (Indonesian to English)
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

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Disease Prediction', 'Symptoms Analysis', 'Food Detection'])
    
    if page == 'Disease Prediction':
        disease_prediction_page()
    elif page == 'Symptoms Analysis':
        symptoms_analysis_page()
    else:
        food_detection_page()

def symptoms_analysis_page():
    st.title("Analisis Gejala")
    
    # Load model and mappings
    model, symptoms_idx, disease_mapping = load_symptoms_model()
    
    # Initialize session state for selected symptoms
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []
    
    # Symptom input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cari gejala:")
        symptom = st.selectbox("", list(symptoms_mapping.keys()))
        if st.button("Tambah Gejala"):
            if symptom not in st.session_state.selected_symptoms:
                st.session_state.selected_symptoms.append(symptom)
    
    with col2:
        st.subheader("Gejala yang dipilih:")
        # Use a temporary list for safe removal while iterating
        symptoms_to_remove = []
        for i, symptom in enumerate(st.session_state.selected_symptoms):
            col_left, col_right = st.columns([6,1])
            with col_left:
                st.write(f"• {symptom}")
            with col_right:
                if st.button("❌", key=f"remove_{i}"):
                    symptoms_to_remove.append(symptom)

        if symptoms_to_remove:
            for symptom in symptoms_to_remove:
                st.session_state.selected_symptoms.remove(symptom)
            st.rerun() # Rerun the script to update the UI instantly

    if st.button("Analisis Penyakit"):
        if len(st.session_state.selected_symptoms) > 0:
            # Convert symptoms to model input format (one-hot encoding)
            input_symptoms = np.zeros(132)
            for symptom in st.session_state.selected_symptoms:
                eng_symptom = symptoms_mapping[symptom]
                if eng_symptom in symptoms_idx:
                    input_symptoms[symptoms_idx[eng_symptom]] = 1

            # Reshape input and make prediction
            input_symptoms = np.array([input_symptoms])
            predictions = model.predict(input_symptoms)
            
            # Get top 3 predictions
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            
            st.subheader("Hasil Analisis:")
            for idx in top_3_idx:
                disease = disease_mapping["idx_to_disease"][str(idx)]
                probability = predictions[0][idx] * 100
                st.write(f"• **{disease}**: {probability:.2f}%")
        else:
            st.warning("Silakan pilih minimal satu gejala terlebih dahulu.")

def predict_multiple_diseases(symptoms_input, threshold=0.1):
    # This function is defined but not used in the main workflow.
    # Implementation would be needed if it were to be used.
    pass

def calculate_derived_features(height, weight, gender, age, blood_pressure, cholesterol, blood_glucose):
    height_m = height / 100
    bmi = weight / (height_m ** 2) if height_m > 0 else 0
    
    # Categories
    bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
    age_category = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
    bp_category = 0 if blood_pressure < 120 else 1 if blood_pressure < 140 else 2 if blood_pressure < 160 else 3
    
    # Interactions
    bmi_age = bmi * age
    bp_age = blood_pressure * age
    bmi_bp = bmi * blood_pressure
    
    # Nutrition calculations
    sodium = weight * 20
    fat = weight * (0.15 if gender == 1 else 0.25) # Assuming gender 1 is male
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
    
    # Input form
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
    
    if st.button('Analisis Kesehatan'):
        # Convert gender to binary (1 for Male, 0 for Female)
        gender_binary = 1 if gender == 'Laki-laki' else 0
        
        # Calculate derived features
        derived = calculate_derived_features(
            height, weight, gender_binary, age,
            blood_pressure, cholesterol, blood_glucose
        )
        
        # Prepare features for the model
        features = np.array([[
            height, weight, gender_binary, age,
            blood_pressure, cholesterol, blood_glucose,
            derived['bmi'], derived['bmi_category'],
            derived['age_category'], derived['bp_category'],
            derived['bmi_age'], derived['bp_age'],
            derived['bmi_bp'], derived['sodium']
        ]])
        
        # Load model and predict
        model = load_disease_model()
        predictions = model.predict(features)
        
        # Display results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader('Hasil Analisis BMI')
            st.write(f"**BMI**: {derived['bmi']:.1f}")
            bmi_status = ('Kurus' if derived['bmi'] < 18.5 else 
                          'Normal' if derived['bmi'] < 25 else 
                          'Gemuk' if derived['bmi'] < 30 else 'Obesitas')
            st.write(f"**Status BMI**: {bmi_status}")
            
            st.subheader('Kebutuhan Nutrisi Harian (Estimasi)')
            st.write(f"Sodium: {derived['sodium']:.1f} mg")
            st.write(f"Lemak: {derived['fat']:.1f} g")
            st.write(f"Protein: {derived['protein']:.1f} g")
            st.write(f"Karbohidrat: {derived['carbs']:.1f} g")
        
        with res_col2:
            st.subheader('Risiko Penyakit')
            diseases = ['Anemia', 'Kolesterol', 'CKD', 'Diabetes', 'Jantung',
                        'Hipertensi', 'MS', 'NAFLD', 'Obesitas', 'Stroke']
            
            # Filter and display diseases with risk > 50%
            high_risk_diseases = []
            for disease, prob in zip(diseases, predictions[0]):
                if prob > 0.5:
                    high_risk_diseases.append(f"• **{disease}** (Risiko: {prob*100:.1f}%)")
            
            if high_risk_diseases:
                for disease_risk in high_risk_diseases:
                    st.write(disease_risk)
            else:
                st.success("Tidak ada risiko penyakit signifikan yang terdeteksi.")

def food_detection_page():
    st.title('Deteksi Makanan dari Gambar')
    st.info("Fitur ini sedang dalam pengembangan.")
    # Provide link to the GitHub repository for more info
    st.write("Untuk informasi lebih lanjut tentang model deteksi makanan, "
             "kunjungi repositori GitHub berikut:")
    st.markdown("[TFoodDetection oleh Wistchze](https://github.com/Wistchze/TFoodDetection)")

@st.cache_resource
def load_food_model():
    """Load the food detection model (currently a placeholder)."""
    # This function is not called but is here for future implementation.
    # It would load the food detection model, e.g., Mask R-CNN.
    return tf.keras.models.load_model('models/model_handi/mrcnn_food_detection.h5')

if __name__ == '__main__':
    main()
