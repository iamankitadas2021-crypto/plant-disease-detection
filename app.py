

import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import json
import os
from PIL import Image
from datetime import datetime

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")
DB_PATH = os.path.join(BASE_DIR, "plant_history.db")

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱")

# Load model and classes
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# Create database
def create_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            disease TEXT,
            confidence REAL,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(image_name, disease, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (image_name, disease, confidence, date) VALUES (?, ?, ?, ?)",
        (image_name, disease, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    data = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return data

create_database()

# Sidebar
st.sidebar.title("🌱 Plant Disease App")
page = st.sidebar.radio(
    "Choose Page",
    ["Home", "Detect Disease", "Prediction History", "Admin Dashboard"]
)

if page == "Home":
    st.title("🌱 Plant Disease Detection")
    st.write("Detect possible plant diseases using Deep Learning.")
    st.info("Upload a leaf image to get a prediction.")
    st.metric("Disease Classes", len(class_names))
    st.write("Model: MobileNetV2")

elif page == "Detect Disease":
    st.title("🔍 Detect Plant Disease")
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("Image must be smaller than 5 MB.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Leaf", use_container_width=True)

            if st.button("Predict Disease"):
                resized = image.resize((128, 128))
                img_array = np.array(resized, dtype=np.float32)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array)
                index = int(np.argmax(predictions[0]))
                disease = class_names[index]
                confidence = float(predictions[0][index]) * 100

                st.success("Prediction: " + disease.replace("___", " - "))
                st.write(f"Confidence: {confidence:.2f}%")

                save_prediction(uploaded_file.name, disease, confidence)

                st.warning("This is an AI prediction, not a confirmed agricultural diagnosis.")

elif page == "Prediction History":
    st.title("📋 Prediction History")
    history = get_history()

    if history:
        st.dataframe(
            [
                {
                    "ID": row[0],
                    "Image": row[1],
                    "Disease": row[2],
                    "Confidence (%)": round(row[3], 2),
                    "Date": row[4]
                }
                for row in history
            ],
            use_container_width=True
        )
    else:
        st.info("No predictions saved yet.")

elif page == "Admin Dashboard":
    st.title("📊 Admin Dashboard")
    history = get_history()

    st.metric("Total Predictions", len(history))

    if history:
        counts = {}
        for row in history:
            counts[row[2]] = counts.get(row[2], 0) + 1

        st.bar_chart(counts)
        st.subheader("Recent Predictions")
        st.dataframe(history, use_container_width=True)
    else:
        st.info("No prediction data available yet.")
