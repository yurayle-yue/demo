import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown # Import library gdown

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
# Fungsi untuk Memuat Model (dengan Caching dan Download)
# =====================================================================================
@st.cache_resource
def load_models():
    """Memuat semua model machine learning dari file."""
    disease_model = None
    symptom_model = None
    classification_model = None
    
    # Pastikan folder 'models' ada
    if not os.path.exists('models'):
        os.makedirs('models')

    # --- Model Prediksi Penyakit ---
    try:
        if os.path.exists('models/disease-prediction-tf-model.h5'):
            disease_model = tf.keras.models.load_model('models/disease-prediction-tf-model.h5')
        else:
            st.warning("File 'disease-prediction-tf-model.h5' tidak ditemukan.")
    except Exception as e:
        st.error(f"Gagal memuat model penyakit: {e}")

    # --- Model Prediksi Gejala ---
    try:
        if os.path.exists('models/symptoms_predict_model.h5'):
            symptom_model = tf.keras.models.load_model('models/symptoms_predict_model.h5')
        else:
            st.warning("File 'symptoms_predict_model.h5' tidak ditemukan.")
    except Exception as e:
        st.error(f"Gagal memuat model gejala: {e}")
            
    # --- Model Klasifikasi Makanan (dengan unduhan dari Google Drive) ---
    tflite_model_path = 'models/bestmodel.tflite'
    gdrive_file_id = '1fRrv1FCg8hd2uLyBUbhB0DpI6teixUob' # ID dari link Google Drive Anda

    try:
        if not os.path.exists(tflite_model_path):
            st.info("Model klasifikasi makanan tidak ditemukan. Mengunduh dari Google Drive...")
            with st.spinner("Proses unduh sedang berjalan, ini mungkin memakan waktu beberapa saat..."):
                gdown.download(id=gdrive_file_id, output=tflite_model_path, quiet=False)
            st.success("Model klasifikasi makanan berhasil diunduh!")

        classification_model = tf.lite.Interpreter(model_path=tflite_model_path)
        classification_model.allocate_tensors()
    except Exception as e:
        st.error(f"Gagal mengunduh atau memuat model klasifikasi makanan: {e}")

    return disease_model, symptom_model, classification_model

# Memuat model saat aplikasi dimulai
disease_model, symptom_model, classification_model = load_models()

# =====================================================================================
# Halaman Aplikasi (Menggunakan Navigasi Sidebar)
# =====================================================================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Selamat Datang", "Prediksi Penyakit", "Analisis Makanan"])

# --- Halaman 1: Selamat Datang ---
if page == "Selamat Datang":
    st.title("Selamat Datang di Tilik Nutrisi 🥗")
    st.markdown("""
    **Solusi cerdas untuk memantau kesehatan dan nutrisi Anda.**

    Aplikasi ini membantu Anda untuk:
    - **Memprediksi Risiko Penyakit**: Berdasarkan data kesehatan dan gejala yang Anda rasakan.
    - **Menganalisis Makanan**: Mengetahui estimasi gizi dari foto makanan Anda.

    Gunakan menu navigasi di sebelah kiri untuk memulai.
    """)
    try:
        image = Image.open('assets/welcome_image.png')
        st.image(image, caption="Ilustrasi oleh AI")
    except FileNotFoundError:
        st.info("Letakkan gambar 'welcome_image.png' di folder 'assets/' untuk tampilan yang lebih menarik.")

# --- Halaman 2: Prediksi Penyakit ---
elif page == "Prediksi Penyakit":
    st.header("🔬 Prediksi Risiko Penyakit")
    st.write("Masukkan data kesehatan dan gejala Anda untuk mendapatkan analisis.")

    with st.form("health_data_form"):
        st.subheader("Data Diri & Kesehatan")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Usia", min_value=1, max_value=120, value=30)
            height = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=160)
            systolic_bp = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=50, max_value=250, value=120)
            fasting_sugar = st.number_input("Gula Darah Puasa (mg/dL)", min_value=50, max_value=500, value=90)
        with col2:
            gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
            weight = st.number_input("Berat Badan (kg)", min_value=10, max_value=200, value=60)
            cholesterol = st.number_input("Kolesterol Total (mg/dL)", min_value=100, max_value=400, value=180)

        st.subheader("Analisis Gejala")
        all_symptoms = ['Sakit kepala', 'Pusing', 'Mual', 'Kelelahan', 'Nyeri dada', 'Sesak napas', 'Sering buang air kecil', 'Haus berlebihan', 'Lapar berlebihan', 'Penglihatan kabur', 'Luka sulit sembuh', 'Kesemutan', 'Kulit pucat', 'Nyeri sendi', 'Demam', 'Pembengkakan kaki', 'Sakit perut', 'Berat badan turun drastis']
        selected_symptoms = st.multiselect("Pilih gejala yang Anda rasakan:", all_symptoms)

        submitted = st.form_submit_button("Analisis Sekarang")

    if submitted:
        if disease_model is not None and symptom_model is not None:
            with st.spinner("Menganalisis data Anda..."):
                gender_numeric = 0 if gender == "Perempuan" else 1
                bmi = weight / ((height / 100) ** 2)
                disease_input = np.array([[age, gender_numeric, bmi, systolic_bp, cholesterol, fasting_sugar]], dtype=np.float32)
                symptom_input = np.zeros((1, len(all_symptoms)), dtype=np.float32)
                for symptom in selected_symptoms:
                    if symptom in all_symptoms:
                        symptom_input[0, all_symptoms.index(symptom)] = 1

                disease_prediction = disease_model.predict(disease_input)
                symptom_prediction = symptom_model.predict(symptom_input)

                st.success("Analisis Selesai!")
                st.subheader("Hasil Prediksi")
                disease_labels = ['Diabetes', 'Hipertensi', 'Penyakit Jantung', 'Stroke', 'Obesitas']
                for i, label in enumerate(disease_labels):
                    prob = disease_prediction[0][i] * 100
                    st.write(f"**{label}**")
                    st.progress(int(prob))
                    st.write(f"Risiko: {prob:.2f}%")
                
                st.write("---")
                st.write("Berdasarkan gejala, kemungkinan Anda mengalami:")
                predicted_symptom_disease_index = np.argmax(symptom_prediction)
                st.info(f"**{disease_labels[predicted_symptom_disease_index]}**")
        else:
            st.error("Satu atau lebih model tidak dapat dimuat. Analisis tidak dapat dilanjutkan.")

# --- Halaman 3: Analisis Makanan ---
elif page == "Analisis Makanan":
    st.header("📸 Analisis Gizi Makanan")
    st.write("Unggah foto makanan Anda untuk mendapatkan klasifikasi dan estimasi nutrisi.")

    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("")

        if st.button("Analisis Gambar Ini"):
            if classification_model is not None:
                with st.spinner("Mengklasifikasikan makanan..."):
                    input_details = classification_model.get_input_details()
                    output_details = classification_model.get_output_details()
                    
                    height = input_details[0]['shape'][1]
                    width = input_details[0]['shape'][2]
                    
                    img_resized = image.resize((width, height))
                    img_array = np.array(img_resized, dtype=np.float32)
                    
                    # Jika model Anda dilatih dengan gambar RGB (3 channel), pastikan gambar juga 3 channel
                    if len(img_array.shape) == 2: # Jika grayscale
                        img_array = np.stack((img_array,)*3, axis=-1)
                    elif img_array.shape[2] == 4: # Jika punya alpha channel (RGBA)
                        img_array = img_array[:, :, :3]

                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    classification_model.set_tensor(input_details[0]['index'], img_array)
                    classification_model.invoke()
                    output_data = classification_model.get_tensor(output_details[0]['index'])
                    
                    # PENTING: Ganti daftar ini dengan nama kelas/label MAKANAN Anda
                    food_labels = ['Nasi Goreng', 'Sate Ayam', 'Bakso', 'Rendang', 'Gado-gado'] 
                    
                    predicted_index = np.argmax(output_data)
                    confidence = output_data[0][predicted_index]
                    predicted_label = food_labels[predicted_index]

                    st.success(f"Gambar berhasil diklasifikasikan!")
                    st.subheader("Hasil Klasifikasi")
                    st.metric(label="Prediksi Makanan", value=predicted_label)
                    st.metric(label="Tingkat Keyakinan", value=f"{confidence*100:.2f}%")

                    food_nutrition_db = {
                        "Nasi Goreng": {"Kalori": 330, "Protein (g)": 10, "Lemak (g)": 15},
                        "Sate Ayam": {"Kalori": 250, "Protein (g)": 25, "Lemak (g)": 14},
                        "Bakso": {"Kalori": 280, "Protein (g)": 15, "Lemak (g)": 18},
                        "Rendang": {"Kalori": 400, "Protein (g)": 23, "Lemak (g)": 28},
                        "Gado-gado": {"Kalori": 350, "Protein (g)": 18, "Lemak (g)": 22}
                    }

                    st.write("---")
                    st.subheader("Estimasi Kandungan Gizi")
                    if predicted_label in food_nutrition_db:
                        nutrition_info = food_nutrition_db[predicted_label]
                        st.json(nutrition_info)
                    else:
                        st.warning(f"Informasi gizi untuk '{predicted_label}' belum tersedia di database.")
                    
                    st.info("**Disclaimer**: Hasil ini adalah estimasi berdasarkan model AI dan mungkin tidak 100% akurat.")
            else:
                st.error("Model klasifikasi makanan tidak dapat dimuat. Analisis tidak dapat dilanjutkan.")
