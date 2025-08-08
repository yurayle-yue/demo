import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

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
# Inisialisasi Session State untuk Menyimpan Data Pengguna
# =====================================================================================
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

# =====================================================================================
# Fungsi untuk Memuat Model (dengan Caching dan Download)
# =====================================================================================
@st.cache_resource
def load_models():
    """Memuat semua model machine learning dari file."""
    disease_model = None
    symptom_model = None
    classification_model = None
    
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- Model Prediksi Penyakit & Gejala ---
    try:
        if os.path.exists('models/disease-prediction-tf-model.h5'):
            disease_model = tf.keras.models.load_model('models/disease-prediction-tf-model.h5')
        if os.path.exists('models/symptoms_predict_model.h5'):
            symptom_model = tf.keras.models.load_model('models/symptoms_predict_model.h5')
    except Exception as e:
        st.error(f"Gagal memuat model penyakit/gejala: {e}")
            
    # --- Model Klasifikasi Makanan (dengan unduhan dari Google Drive) ---
    tflite_model_path = 'models/bestmodel.tflite'
    gdrive_file_id = '1fRrv1FCg8hd2uLyBUbhB0DpI6teixUob'

    try:
        if not os.path.exists(tflite_model_path):
            with st.spinner(f"Mengunduh model klasifikasi makanan dari Google Drive..."):
                gdown.download(id=gdrive_file_id, output=tflite_model_path, quiet=False)
        
        classification_model = tf.lite.Interpreter(model_path=tflite_model_path)
        classification_model.allocate_tensors()
    except Exception as e:
        st.error(f"Gagal mengunduh atau memuat model klasifikasi makanan: {e}")

    return disease_model, symptom_model, classification_model

# Memuat model saat aplikasi dimulai
disease_model, symptom_model, classification_model = load_models()

# =====================================================================================
# Halaman Aplikasi
# =====================================================================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Selamat Datang", "Prediksi Penyakit", "Prediksi Gejala", "Analisis Makanan"])

# --- Halaman 1: Selamat Datang ---
if page == "Selamat Datang":
    st.title("Selamat Datang di Tilik Nutrisi 🥗")
    st.markdown("""
    **Solusi cerdas untuk memantau kesehatan dan nutrisi Anda.**
    
    Gunakan menu navigasi di sebelah kiri untuk memulai.
    """)
    
    # Menampilkan ringkasan profil jika sudah ada
    if st.session_state.user_profile:
        st.subheader("Ringkasan Profil Kesehatan Anda")
        profile = st.session_state.user_profile
        
        if 'height' in profile and 'weight' in profile:
            bmi = profile['weight'] / ((profile['height'] / 100) ** 2)
            st.metric(label="BMI (Indeks Massa Tubuh)", value=f"{bmi:.1f}")
        
        # Menampilkan risiko tertinggi dari data kesehatan
        if 'predictions_health' in profile:
            top_risk_label, top_risk_prob = max(profile['predictions_health'].items(), key=lambda item: item[1])
            st.metric(label="Risiko Penyakit Tertinggi (dari data kesehatan)", value=top_risk_label, delta=f"{(top_risk_prob*100):.1f}% Risiko", delta_color="inverse")
        
        # Menampilkan prediksi dari gejala
        if 'prediction_symptom' in profile:
             st.metric(label="Prediksi Penyakit (dari gejala)", value=profile['prediction_symptom'])
    else:
        st.info("Anda belum melakukan pemeriksaan kesehatan. Silakan gunakan menu navigasi untuk memulai.")

    try:
        image = Image.open('assets/welcome_image.png')
        st.image(image, caption="Ilustrasi oleh AI")
    except FileNotFoundError:
        pass

# --- Halaman 2: Prediksi Penyakit ---
elif page == "Prediksi Penyakit":
    st.header("🔬 Prediksi Risiko Penyakit")
    st.write("Masukkan data kesehatan Anda untuk mendapatkan analisis risiko.")

    with st.form("health_data_form"):
        st.subheader("Data Diri & Kesehatan")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia", 1, 120, 30)
            height = st.number_input("Tinggi Badan (cm)", 50, 250, 160)
            systolic_bp = st.number_input("Tekanan Darah Sistolik (mmHg)", 50, 250, 120)
            fasting_sugar = st.number_input("Gula Darah Puasa (mg/dL)", 50, 500, 90)
        with col2:
            gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
            weight = st.number_input("Berat Badan (kg)", 10, 200, 60)
            cholesterol = st.number_input("Kolesterol Total (mg/dL)", 100, 400, 180)
        
        submitted = st.form_submit_button("Analisis & Simpan Profil")

    if submitted:
        if disease_model is not None:
            with st.spinner("Menganalisis data Anda..."):
                gender_numeric = 0 if gender == "Perempuan" else 1
                bmi = weight / ((height / 100) ** 2)
                health_input = np.array([[age, gender_numeric, bmi, systolic_bp, cholesterol, fasting_sugar]], dtype=np.float32)
                
                health_prediction_tensor = disease_model(health_input, training=False)
                health_prediction = health_prediction_tensor[0].numpy()
                
                disease_labels = ['Diabetes', 'Hipertensi', 'Penyakit Jantung', 'Stroke', 'Obesitas']
                predictions_health_dict = {label: float(prob) for label, prob in zip(disease_labels, health_prediction)}

                # Inisialisasi atau update session state
                if st.session_state.user_profile is None:
                    st.session_state.user_profile = {}
                
                # Menyimpan data kesehatan ke session state
                st.session_state.user_profile.update({
                    "age": age, "height": height, "weight": weight, "gender": gender,
                    "systolic_bp": systolic_bp, "cholesterol": cholesterol,
                    "fasting_sugar": fasting_sugar,
                    "predictions_health": predictions_health_dict
                })
                
                st.success("Analisis Selesai! Profil kesehatan Anda telah disimpan untuk sesi ini.")
                
                st.subheader("Hasil Prediksi dari Data Kesehatan")
                for label, prob in predictions_health_dict.items():
                    st.write(f"**{label}**")
                    st.progress(int(prob * 100))
                    st.write(f"Risiko: {prob*100:.2f}%")
        else:
            st.error("Model prediksi penyakit tidak dapat dimuat.")

# --- Halaman 3: Prediksi Gejala ---
elif page == "Prediksi Gejala":
    st.header("🩺 Prediksi Penyakit Berdasarkan Gejala")
    st.write("Pilih gejala yang Anda rasakan untuk mendapatkan prediksi.")

    with st.form("symptom_data_form"):
        all_symptoms = ['Sakit kepala', 'Pusing', 'Mual', 'Kelelahan', 'Nyeri dada', 'Sesak napas', 'Sering buang air kecil', 'Haus berlebihan', 'Lapar berlebihan', 'Penglihatan kabur', 'Luka sulit sembuh', 'Kesemutan', 'Kulit pucat', 'Nyeri sendi', 'Demam', 'Pembengkakan kaki', 'Sakit perut', 'Berat badan turun drastis']
        selected_symptoms = st.multiselect("Pilih gejala yang Anda rasakan:", all_symptoms)
        
        submitted_symptoms = st.form_submit_button("Analisis Gejala")

    if submitted_symptoms:
        if symptom_model is not None:
            with st.spinner("Menganalisis gejala Anda..."):
                symptom_input = np.zeros((1, len(all_symptoms)), dtype=np.float32)
                for symptom in selected_symptoms:
                    if symptom in all_symptoms:
                        symptom_input[0, all_symptoms.index(symptom)] = 1
                
                symptom_prediction_tensor = symptom_model(symptom_input, training=False)
                symptom_prediction_index = np.argmax(symptom_prediction_tensor[0].numpy())
                
                disease_labels = ['Diabetes', 'Hipertensi', 'Penyakit Jantung', 'Stroke', 'Obesitas'] # Asumsi labelnya sama
                predicted_symptom_label = disease_labels[symptom_prediction_index]

                # Inisialisasi atau update session state
                if st.session_state.user_profile is None:
                    st.session_state.user_profile = {}

                # Menyimpan hasil prediksi gejala ke session state
                st.session_state.user_profile.update({
                    "symptoms": selected_symptoms,
                    "prediction_symptom": predicted_symptom_label
                })

                st.success("Analisis Gejala Selesai!")
                st.subheader("Hasil Prediksi dari Gejala")
                st.info(f"Berdasarkan gejala yang Anda pilih, kemungkinan penyakit yang diderita adalah: **{predicted_symptom_label}**")
        else:
            st.error("Model prediksi gejala tidak dapat dimuat.")

# --- Halaman 4: Analisis Makanan ---
elif page == "Analisis Makanan":
    st.header("📸 Analisis Gizi Makanan")
    
    if not st.session_state.user_profile or 'predictions_health' not in st.session_state.user_profile:
        st.warning("Harap lakukan 'Prediksi Penyakit' terlebih dahulu untuk mendapatkan rekomendasi makanan yang personal.")
        st.stop()

    st.write("Unggah foto makanan Anda untuk mendapatkan klasifikasi dan estimasi nutrisi.")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        if st.button("Analisis Gambar Ini"):
            if classification_model is not None:
                with st.spinner("Mengklasifikasikan makanan..."):
                    input_details = classification_model.get_input_details()
                    output_details = classification_model.get_output_details()
                    height = input_details[0]['shape'][1]
                    width = input_details[0]['shape'][2]
                    img_resized = image.resize((width, height))
                    img_array = np.array(img_resized, dtype=np.float32)
                    if len(img_array.shape) == 2: img_array = np.stack((img_array,)*3, axis=-1)
                    if img_array.shape[2] == 4: img_array = img_array[:, :, :3]
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    classification_model.set_tensor(input_details[0]['index'], img_array)
                    classification_model.invoke()
                    output_data = classification_model.get_tensor(output_details[0]['index'])
                    
                    food_labels = ['Nasi Goreng', 'Sate Ayam', 'Bakso', 'Rendang', 'Gado-gado'] 
                    predicted_index = np.argmax(output_data)
                    predicted_label = food_labels[predicted_index]

                    st.success(f"Gambar berhasil diklasifikasikan sebagai: **{predicted_label}**")
                    
                    st.subheader("Rekomendasi Personal Untuk Anda")
                    profile = st.session_state.user_profile
                    if predicted_label in ["Rendang", "Nasi Goreng"] and profile['predictions_health']['Hipertensi'] > 0.5:
                        st.error("Makanan ini cenderung tinggi lemak dan sodium. Kurang disarankan untuk Anda yang berisiko Hipertensi.")
                    elif predicted_label == "Nasi Goreng" and profile['predictions_health']['Diabetes'] > 0.6:
                        st.warning("Porsi nasi yang banyak dapat meningkatkan gula darah. Pertimbangkan porsi yang lebih kecil.")
                    else:
                        st.success("Makanan ini terlihat cukup sesuai dengan profil kesehatan Anda saat ini. Nikmati secukupnya!")
            else:
                st.error("Model klasifikasi makanan tidak dapat dimuat.")
