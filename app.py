import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tempfile

# ---------------- Configuration ----------------
MODEL_PATH = r"C:\Users\ASUS\Desktop\smart_waste_management\efficientnet_trashnet.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- Image Transform ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- Initialize Stats ----------------
stats = Counter({cls: 0 for cls in CLASS_NAMES})

# ---------------- Streamlit App ----------------
st.title("🗑️ Smart Waste Classification System")
st.write("Upload images or use your webcam to classify waste items in real-time.")

# ---------------- Image Upload Mode ----------------
st.header("📁 Image Upload")
uploaded_files = st.file_uploader("Upload one or more images", type=["jpg","jpeg","png"], accept_multiple_files=True)

def classify_image(image):
    img = Image.open(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        category = CLASS_NAMES[predicted.item()]
    return category

if uploaded_files:
    for uploaded_file in uploaded_files:
        category = classify_image(uploaded_file)
        stats[category] += 1
        st.image(uploaded_file, caption=f"Predicted Category: {category}", use_column_width=True)
        st.write(f"Item classified and sorted into category: **{category}**")

# ---------------- Webcam Mode ----------------
st.header("📷 Webcam Classification")
use_webcam = st.checkbox("Enable webcam for real-time classification")

if use_webcam:
    st.write("Click 'Start' to begin webcam classification.")
    start_btn = st.button("Start Webcam")
    stop_btn = st.button("Stop Webcam")

    FRAME_WINDOW = st.image([])  # Placeholder for frames

    if start_btn:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame from webcam.")
                break

            # Convert frame to PIL Image for model
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                category = CLASS_NAMES[predicted.item()]
                stats[category] += 1

            # Display category on frame
            cv2.putText(frame, f"Category: {category}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Stop condition
            if stop_btn:
                break

        cap.release()
        cv2.destroyAllWindows()

# ---------------- Display Statistics ----------------
st.header("📊 Classification Statistics")
df_stats = pd.DataFrame.from_dict(stats, orient='index', columns=['Count'])
st.dataframe(df_stats)

# ---------------- Bar Chart ----------------
st.subheader("📈 Visualization")
fig, ax = plt.subplots()
ax.bar(stats.keys(), stats.values(), color='green')
ax.set_xlabel("Waste Categories")
ax.set_ylabel("Number of Items")
ax.set_title("Waste Classification Statistics")
st.pyplot(fig)

# ---------------- Save CSV Report ----------------
df_stats.to_csv("waste_classification_report.csv")
st.write("✅ Statistical report saved as `waste_classification_report.csv`")
