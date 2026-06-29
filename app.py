# ============================================================
# 🍌 BananaVision AI Pro
# Smart Banana Ripeness Classification
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BananaVision AI",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

LOGO_PATH = "assets/logo.png"
BANNER_PATH = "assets/banner.png"
HISTORY_FILE = "prediction_history.csv"

# ============================================================
# CLASS LABELS
# ============================================================

CLASS_NAMES = ["Overripe", "Ripe", "Rotten", "Unripe"]

IMG_SIZE = (224, 224)

# ============================================================
# MULTI MODEL PATHS  ← moved to TOP so model loads correctly
# ============================================================

MODEL_PATHS = {
    "CNN":           "models/banana_ripeness_cnn.keras",
    "MobileNetV2":   "models/mobilenetv2.keras",
    "EfficientNetB0":"models/efficientnetb0.keras",
    "ResNet50":      "models/resnet50.keras"
}

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)

    st.markdown("## 🍌 BananaVision AI")
    st.success("AI Powered Fruit Quality Detection")
    st.markdown("---")

    st.subheader("📋 Project Information")
    st.write("✔ Deep Learning")
    st.write("✔ CNN Model")
    st.write("✔ TensorFlow")
    st.write("✔ Streamlit")
    st.write("✔ Computer Vision")
    st.markdown("---")

    st.subheader("📊 Model Details")
    st.info(f"""
Model : CNN

Classes : {len(CLASS_NAMES)}

Image Size : {IMG_SIZE[0]} x {IMG_SIZE[1]}

Framework : TensorFlow
""")
    st.markdown("---")

    st.subheader("🍌 Banana Classes")
    st.write("🟤 Overripe")
    st.write("🟡 Ripe")
    st.write("⚫ Rotten")
    st.write("🟢 Unripe")
    st.markdown("---")

    # ── Model selector ──────────────────────────────────────
    st.subheader("🧠 AI Model Selection")
    selected_model = st.selectbox(
        "Choose Deep Learning Model",
        list(MODEL_PATHS.keys())
    )
    st.markdown("---")

    st.subheader("👨‍💻 Developer")
    st.write("**Winston Raj H**")
    st.caption("AI & Data Science Project")
    st.markdown("---")
    st.success("BananaVision AI Pro Version 1.0")

# ============================================================
# LOAD SELECTED MODEL
# ============================================================

@st.cache_resource
def load_selected_model(model_name):
    return load_model(MODEL_PATHS[model_name])

model = load_selected_model(selected_model)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main { background-color: #F8F9FA; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1 { color: #1B4332; font-weight: bold; text-align: center; }
h2 { color: #2D6A4F; }
h3 { color: #40916C; }
.stButton>button {
    background: #2D6A4F; color: white; border-radius: 10px;
    height: 50px; width: 100%; font-size: 18px;
    font-weight: bold; border: none;
}
.stButton>button:hover { background: #1B4332; }
.stFileUploader { border: 2px dashed #52B788; padding: 15px; border-radius: 10px; }
.footer { text-align: center; padding: 15px; font-size: 15px; color: gray; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PAGE TITLE
# ============================================================

st.title("🍌 BananaVision AI Pro")
st.caption("AI Powered Banana Ripeness Classification System")
st.divider()

# ============================================================
# MAIN PAGE INTRO
# ============================================================

if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, use_container_width=True)

st.markdown("## 🍌 Welcome to BananaVision AI")
st.write("""
BananaVision AI is an Artificial Intelligence based Banana Ripeness
Classification System. The model predicts banana quality into one of four categories.
""")

col1, col2 = st.columns(2)
with col1:
    st.info("""
### 🎯 Features
- Upload Banana Image
- CNN Deep Learning Model
- AI Prediction
- Confidence Score
- Professional Dashboard
- Easy to Use
""")
with col2:
    st.success("""
### 📌 Technologies Used
- Python
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- Pillow
""")

st.markdown("---")

# ============================================================
# FILE UPLOADER
# ============================================================

st.subheader("📤 Upload Banana Image")
uploaded_file = st.file_uploader(
    "Choose a Banana Image",
    type=["jpg", "jpeg", "png"]
)

st.markdown("---")
st.markdown("""
<div class='footer'>Made with ❤️ using Streamlit &amp; TensorFlow</div>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_image(pil_image):
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(IMG_SIZE)
    arr = np.array(pil_image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_banana(pil_image):
    processed = preprocess_image(pil_image)
    prediction = model.predict(processed, verbose=0)   # shape (1, 4)
    probabilities = prediction[0]                       # shape (4,)
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = CLASS_NAMES[predicted_index]
    return predicted_class, confidence, probabilities

# ============================================================
# PREDICTION BLOCK — everything that needs results lives here
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # ── Run prediction ──────────────────────────────────────
    predicted_class, confidence, probabilities = predict_banana(image)
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # ── Image + top metrics ─────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Banana", use_container_width=True)
    with col2:
        st.success("✅ Prediction Completed Successfully")
        st.metric("Predicted Class", predicted_class)
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(float(confidence))

    # ── Class Probabilities ─────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction Probability")
    for i in range(len(CLASS_NAMES)):
        st.write(f"{CLASS_NAMES[i]} : {probabilities[i] * 100:.2f}%")
        st.progress(float(probabilities[i]))   # ← FIX: no redundant model.predict

    # ── AI Recommendation ───────────────────────────────────
    st.markdown("---")
    st.subheader("💡 Recommendation")
    if predicted_class == "Unripe":
        st.info("• Not Ready to Eat\n• Store for Few Days\n• Good for Transportation")
    elif predicted_class == "Ripe":
        st.success("• Ready to Eat\n• Sweet Taste\n• Best Quality")
    elif predicted_class == "Overripe":
        st.warning("• Consume Immediately\n• Suitable for Juice\n• Suitable for Banana Cake")
    else:
        st.error("• Rotten Banana\n• Not Safe to Eat\n• Please Discard")

    # ── Confidence Bar Chart ────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Prediction Confidence")
    confidence_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Confidence": probabilities
    })
    st.bar_chart(confidence_df.set_index("Class"))

    # ── Download Text Report ────────────────────────────────
    st.markdown("---")
    report = f"""
====================================
BANANAVISION AI REPORT
====================================
Prediction : {predicted_class}
Confidence : {confidence * 100:.2f}%
------------------------------------
Class Probabilities
"""
    for i in range(len(CLASS_NAMES)):
        report += f"{CLASS_NAMES[i]} : {probabilities[i] * 100:.2f}%\n"
    report += """
====================================
Generated by BananaVision AI
====================================
"""
    st.download_button(
        label="📄 Download Prediction Report",
        data=report,
        file_name="BananaVision_Report.txt",
        mime="text/plain"
    )

    # ── Smart Recommendation (Part 5) ───────────────────────
    st.markdown("---")
    st.header("🤖 AI Smart Recommendation")
    st.caption(f"Prediction Time : {current_time}")
    st.markdown("### 🍌 Banana Usage Recommendation")
    if predicted_class == "Unripe":
        st.info("✅ Not Ready to Eat\n• Keep for 2–4 days at room temperature\n• Good for transportation\n• Suitable for export")
    elif predicted_class == "Ripe":
        st.success("✅ Ready to Eat\n• Sweet Taste\n• High Nutrition\n• Best for Direct Consumption\n• Suitable for Sale")
    elif predicted_class == "Overripe":
        st.warning("⚠ Overripe Banana\n🍰 Banana Cake\n🥤 Banana Shake\n🍌 Smoothie\n🍞 Banana Bread")
    else:
        st.error("❌ Rotten Banana\n• Do not consume\n• Dispose safely\n• Can be used for compost")

    # ── Analytics Dashboard (Part 7) ────────────────────────
    st.markdown("---")
    st.header("📈 BananaVision AI Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🍌 Prediction", predicted_class)
    with col2:
        st.metric("🎯 Confidence", f"{confidence * 100:.2f}%")
    with col3:
        st.metric("🧠 CNN Classes", len(CLASS_NAMES))
    with col4:
        st.metric("📸 Test Images", "562")

    prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probabilities})
    st.markdown("---")
    st.subheader("📊 Confidence Chart")
    st.bar_chart(prob_df.set_index("Class"))

    st.markdown("---")
    st.subheader("🥧 Prediction Distribution")
    pie_df = prob_df.copy()
    pie_df["Probability"] = pie_df["Probability"] * 100
    st.dataframe(pie_df, use_container_width=True)

    highest = pie_df.loc[pie_df["Probability"].idxmax()]
    st.success(f"🏆 Highest Confidence Class\n\nClass : {highest['Class']}\n\nConfidence : {highest['Probability']:.2f}%")

    st.markdown("---")
    st.subheader("📝 Prediction Summary")
    summary = f"""
Prediction  : {predicted_class}
Confidence  : {confidence * 100:.2f}%
Model       : {selected_model}
Time        : {current_time}
CNN classified this image successfully.
"""
    st.code(summary)

    st.markdown("---")
    st.subheader("💻 Session Details")
    st.json({
        "Framework": "TensorFlow",
        "Application": "Streamlit",
        "Model": selected_model,
        "Image Size": "224x224",
        "Accuracy": "95.91%"
    })

    # ── Prediction History (Part 8) ──────────────────────────
    st.markdown("---")
    st.header("📋 Prediction History")

    new_row = pd.DataFrame({
        "Date":       [datetime.now().strftime("%d-%m-%Y")],
        "Time":       [datetime.now().strftime("%H:%M:%S")],
        "Prediction": [predicted_class],
        "Confidence": [round(confidence * 100, 2)]
    })

    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        history_df = pd.concat([history_df, new_row], ignore_index=True)
    else:
        history_df = new_row

    history_df.to_csv(HISTORY_FILE, index=False)
    st.dataframe(history_df, use_container_width=True)
    st.download_button(
        label="⬇ Download Prediction History",
        data=history_df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    # ── Interactive Plotly Dashboard (Part 9) ────────────────
    st.markdown("---")
    st.header("📊 Interactive AI Dashboard")

    chart_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Confidence": probabilities * 100   # ← numpy array * 100 is fine
    })

    fig_bar = px.bar(
        chart_df, x="Class", y="Confidence",
        color="Confidence", text="Confidence",
        title="Prediction Confidence"
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig_bar.update_layout(height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_pie = px.pie(
        chart_df, names="Class", values="Confidence",
        title="Prediction Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    best = chart_df.loc[chart_df["Confidence"].idxmax()]
    st.success(f"🏆 Highest Confidence\n\nClass : **{best['Class']}**\n\nConfidence : **{best['Confidence']:.2f}%**")

    # ── Current Prediction JSON ──────────────────────────────
    st.markdown("---")
    st.header("📋 Current Prediction")
    st.json({
        "Prediction": predicted_class,
        "Confidence": f"{confidence * 100:.2f}%",
        "Time": current_time
    })

# ── Show a prompt if no image uploaded yet ─────────────────
else:
    st.info("👆 Please upload a banana image above to get started.")

# ============================================================
# CAMERA SUPPORT (Part 10)
# ============================================================

st.markdown("---")
st.header("📷 Live Camera Prediction")
camera_image = st.camera_input("Take a Banana Photo")

if camera_image is not None:
    cam_pil = Image.open(camera_image)
    cam_class, cam_conf, cam_probs = predict_banana(cam_pil)

    st.success(f"✅ Camera Prediction : **{cam_class}** ({cam_conf * 100:.2f}%)")

    cam_df = pd.DataFrame({"Class": CLASS_NAMES, "Confidence": cam_probs * 100})
    fig_cam = px.bar(cam_df, x="Class", y="Confidence",
                     color="Confidence", title="Camera Prediction Confidence")
    st.plotly_chart(fig_cam, use_container_width=True)

# ============================================================
# STATIC SECTIONS (no prediction needed)
# ============================================================

st.markdown("---")
st.header("📊 AI Dashboard")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="🎯 Model Accuracy", value="95.91%", delta="+2.3%")
with col2:
    st.metric(label="📸 Images Trained", value="11,793")
with col3:
    st.metric(label="🍌 Classes", value="4")
with col4:
    st.metric(label="🧠 Model", value="CNN")

st.markdown("---")
st.subheader("📚 Dataset Information")
st.json({
    "Training Images": 11793,
    "Validation Images": 1123,
    "Testing Images": 562,
    "Total Classes": 4,
    "Image Size": "224 x 224",
    "Framework": "TensorFlow"
})

st.markdown("---")
st.header("📖 About BananaVision AI")
st.write("""
BananaVision AI is a Deep Learning based Computer Vision application
developed for automatic Banana Ripeness Classification using CNNs.
""")

st.markdown("---")
st.subheader("🛠 Technologies Used")
tech1, tech2, tech3 = st.columns(3)
with tech1:
    st.success("Python\nTensorFlow\nNumPy")
with tech2:
    st.info("Streamlit\nOpenCV\nPillow")
with tech3:
    st.warning("Deep Learning\nCNN\nComputer Vision")

st.markdown("---")
st.header("🧠 AI Model Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Model Name", selected_model)
    st.metric("Input Size", "224 × 224")
    st.metric("Framework", "TensorFlow")
with col2:
    st.metric("Classes", "4")
    st.metric("Training Images", "11,793")
    st.metric("Accuracy", "95.91 %")

st.markdown("---")
st.header("📚 Banana Ripeness Guide")
st.table({
    "🟢 Unripe":  "Green colour, hard texture, not ready to eat.",
    "🟡 Ripe":    "Yellow colour, sweet taste, ready for consumption.",
    "🟤 Overripe":"Dark spots appear, very sweet, useful for baking.",
    "⚫ Rotten":  "Black colour, spoiled, not safe for eating."
})

st.markdown("---")
st.header("🏆 Project Achievements")
a1, a2, a3 = st.columns(3)
with a1:
    st.success("### 🎯 Accuracy\n95.91%\nCNN Model")
with a2:
    st.info("### 📸 Dataset\n13,478 Images\n4 Classes")
with a3:
    st.warning("### ⚡ Framework\nTensorFlow\nStreamlit")

st.markdown("---")
st.header("📞 Developer Information")
col1, col2 = st.columns(2)
with col1:
    st.write("👨‍💻 **Developer** : Winston Raj H")
    st.write("🎓 Data Science")
    st.write("📍 Tamil Nadu, India")
with col2:
    st.write("🛠 Skills")
    for skill in ["Python", "SQL", "Machine Learning", "Deep Learning", "Streamlit"]:
        st.write(f"✔ {skill}")

st.markdown("---")
st.header("🚀 Key Features")
for feature in [
    "🍌 Banana Ripeness Classification",
    "🤖 CNN Deep Learning Model",
    "📷 Image Upload & Camera Support",
    "📊 Confidence Score",
    "📋 Download Prediction Report",
    "📈 Plotly Interactive Charts",
    "🍰 Smart Recommendation",
    "💻 Professional Dashboard"
]:
    st.write(feature)

st.markdown("---")
st.info("""
📌 Disclaimer

This project was developed for educational and research purposes.
The prediction is AI-assisted and should not replace professional quality assessment.
""")

st.success("""
🎉 Thank you for using BananaVision AI

✔ Python  ✔ TensorFlow  ✔ Streamlit  ✔ Deep Learning  ✔ Computer Vision
""")

st.markdown("""
<hr>
<div style="text-align:center">
<h3>🍌 BananaVision AI Pro</h3>
<p>Artificial Intelligence Powered Banana Ripeness Classification</p>
<p>Developed by <b>Winston Raj H</b></p>
<p>TensorFlow • Streamlit • Deep Learning • Computer Vision</p>
<p>Version 1.0</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# END OF APPLICATION
# ============================================================