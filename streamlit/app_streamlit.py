# tilik_nutrisi.py
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import gdown

# ====== Konfigurasi App ======
st.set_page_config(page_title="Tilik Nutrisi", page_icon="🥗", layout="wide")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mrcnn_food_detection.h5")
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1sVzG18PNoUrwxctxIMIoQL79bCWjhnSY&export=download"
USER_DB = "users.json"

os.makedirs(MODEL_DIR, exist_ok=True)

# ====== Helper ======
def download_food_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model deteksi makanan..."):
            gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_food_model():
    download_food_model()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def login_user(email, password):
    users = load_users()
    return users.get(email) and users[email]["password"] == password

def register_user(username, email, password):
    users = load_users()
    if email in users:
        return False
    users[email] = {"username": username, "password": password}
    save_users(users)
    return True

# ====== Halaman ======
def page_welcome():
    st.title("🥗 Tilik Nutrisi")
    st.subheader("Smart Solution for Disease Detection and Food/Nutrition Recommendation")
    st.markdown("""
    Aplikasi ini membantu Anda memantau kesehatan, mendeteksi potensi penyakit,
    dan memberikan rekomendasi makanan sesuai kebutuhan nutrisi Anda.
    """)

def page_signup():
    st.title("🔐 Sign Up")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Daftar"):
        if register_user(username, email, password):
            st.success("Pendaftaran berhasil! Silakan login.")
        else:
            st.error("Email sudah terdaftar.")

def page_login():
    st.title("🔑 Sign In")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(email, password):
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.success("Login berhasil!")
        else:
            st.error("Email atau password salah.")

def page_disease_detection():
    st.title("🩺 Disease Detection")
    height = st.number_input("Tinggi Badan (cm)", min_value=0, value=170)
    weight = st.number_input("Berat Badan (kg)", min_value=0, value=65)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Usia", min_value=0, value=25)
    blood_pressure = st.number_input("Tekanan Darah Sistolik", min_value=0, value=120)
    cholesterol = st.number_input("Kolesterol Total", min_value=0, value=180)
    blood_glucose = st.number_input("Gula Darah Puasa", min_value=0, value=90)

    if st.button("Analisis"):
        bmi = weight / ((height/100)**2)
        st.write(f"**BMI Anda:** {bmi:.1f}")
        st.info("Model prediksi penyakit dapat diintegrasikan di sini.")

def page_food_detection():
    st.title("🍲 Food Detection")
    st.write("Ambil gambar dari kamera atau unggah dari galeri.")

    # Input kamera
    camera_img = st.camera_input("Ambil foto makanan")
    upload_img = st.file_uploader("Atau unggah gambar", type=["jpg", "jpeg", "png"])

    img = None
    if camera_img:
        img = Image.open(camera_img)
    elif upload_img:
        img = Image.open(upload_img)

    if img is not None:
        st.image(img, caption="Gambar yang dipilih", use_column_width=True)
        if st.button("Deteksi Makanan"):
            model = load_food_model()
            img_resized = img.resize((224, 224))
            arr = np.array(img_resized).astype(np.float32)/255.0
            x = np.expand_dims(arr, axis=0)
            try:
                preds = model.predict(x)
                st.success(f"Prediksi berhasil. Output shape: {np.array(preds).shape}")
            except Exception as e:
                st.error(f"Error prediksi: {e}")

def page_nutrition_recommendation():
    st.title("🥦 Nutrition Recommendation")
    st.info("Halaman ini akan memberikan rekomendasi makanan sesuai hasil deteksi penyakit & makanan.")
    st.warning("Fitur ini masih dalam pengembangan.")

# ====== Navigasi ======
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

menu = ["Welcome", "Sign Up", "Sign In", "Disease Detection", "Food Detection", "Nutrition Recommendation"]
choice = st.sidebar.radio("Pilih Halaman", menu)

if choice == "Welcome":
    page_welcome()
elif choice == "Sign Up":
    page_signup()
elif choice == "Sign In":
    page_login()
elif choice == "Disease Detection":
    if st.session_state.logged_in:
        page_disease_detection()
    else:
        st.warning("Silakan login terlebih dahulu.")
elif choice == "Food Detection":
    if st.session_state.logged_in:
        page_food_detection()
    else:
        st.warning("Silakan login terlebih dahulu.")
elif choice == "Nutrition Recommendation":
    if st.session_state.logged_in:
        page_nutrition_recommendation()
    else:
        st.warning("Silakan login terlebih dahulu.")
