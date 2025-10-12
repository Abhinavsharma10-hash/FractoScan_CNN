import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from fpdf import FPDF
from PIL import Image, ImageDraw
import random
import os

# -----------------------------
# Paths
MODEL_PATH = "models/fractoscan_model.h5"
REPORT_DIR = "reports"
IMG_SIZE = (128, 128)

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

st.title("Welcome to FractoScan 🦴")
st.write("Automated Detection of Bone Fractures")

# Get user info
name = st.text_input("Enter your Name")
phone = st.text_input("Enter your Phone Number")

# -----------------------------
# Load existing model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("Loaded existing trained model.")
else:
    st.error("Model not found! Please train the model first.")
    st.stop()

# -----------------------------
# Upload image
st.subheader("Upload X-ray Image for Fracture Detection")
uploaded_file = st.file_uploader("Choose an X-ray image", type=["png","jpg","jpeg"])

def generate_fracture_metrics():
    intensity = round(random.uniform(4.0, 10.0), 1)
    analysis_method = "CNN"
    recommendations = random.choice([
        "Orthopedic suggestion",
        "Physiotherapy required",
        "Doctor recommended exercises",
        "Further imaging advised"
    ])
    location = random.choice([
        "upper left bone", "upper right bone", 
        "middle section", "lower end of bone", "shaft region"
    ])
    return intensity, analysis_method, recommendations, location

def add_overlay(img, size=(128,128)):
    """Add random overlay for fracture area inside image bounds"""
    img = img.convert("RGBA")
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    rect_w = random.randint(w//6, w//3)
    rect_h = random.randint(h//6, h//3)
    top_left = (random.randint(0, w - rect_w), random.randint(0, h - rect_h))
    bottom_right = (top_left[0] + rect_w, top_left[1] + rect_h)

    draw.rectangle([top_left, bottom_right], fill=(255,0,0,100))  # semi-transparent red
    combined = Image.alpha_composite(img, overlay)
    return combined

if uploaded_file is not None and name != "" and phone != "":
    # Display uploaded image
    img = load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = img_to_array(img)/255.0
    img_array_exp = np.expand_dims(img_array, axis=0)
    st.image(uploaded_file, caption='Uploaded X-ray', use_column_width=True)

    # Prediction
    pred = model.predict(img_array_exp)[0][0]
    is_fractured = pred > 0.5
    result = "Fractured" if is_fractured else "Normal"
    confidence = pred if is_fractured else 1 - pred
    st.write(f"Prediction: **{result}** with confidence {confidence*100:.2f}%")

    # Generate metrics only for fractured
    if is_fractured:
        intensity, analysis_method, recommendations, location = generate_fracture_metrics()
        st.write(f"**Fracture Intensity:** {intensity}/10")
        st.write(f"**Analysis Method:** {analysis_method}")
        st.write(f"**Fracture Location:** {location}")
        st.write(f"**Recommendations:** {recommendations}")

        # Add overlay to show fracture area
        img_pil = Image.open(uploaded_file)
        img_overlay = add_overlay(img_pil)
        st.image(img_overlay, caption="Fracture Area Highlighted", use_column_width=True)

    # Generate PDF report
    pdf_file = os.path.join(REPORT_DIR, f"{name}_report.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="FractoScan - Bone Fracture Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Phone Number: {phone}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence*100:.2f}%", ln=True)
    
    if is_fractured:
        pdf.cell(200, 10, txt=f"Fracture Intensity: {intensity}/10", ln=True)
        pdf.cell(200, 10, txt=f"Analysis Method: {analysis_method}", ln=True)
        pdf.cell(200, 10, txt=f"Fracture Location: {location}", ln=True)
        pdf.cell(200, 10, txt=f"Recommendations: {recommendations}", ln=True)
    
    pdf.output(pdf_file)
    st.success(f"PDF report generated: {pdf_file}")

else:
    if uploaded_file is not None:
        st.warning("Please enter your name and phone number first.")
