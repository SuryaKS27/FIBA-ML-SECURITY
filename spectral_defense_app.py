import streamlit as st
import torch
import joblib
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function Definitions
def calculate_fourier_spectrum(im):
    im = im.astype(np.float32)
    fft = np.fft.fft2(im, axes=(0, 1))
    mfs = np.abs(fft)
    return mfs

def spectral_defense_filter(image, clf, threshold=0.5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).numpy().transpose(1, 2, 0)
    mfs = calculate_fourier_spectrum(image).flatten().reshape(1, -1)
    probability = clf.predict_proba(mfs)[0][1]
    return probability > threshold

def spectral_defense_pipeline(image, spectral_model_path, fiba_model, threshold=0.5):
    clf = joblib.load(spectral_model_path)
    is_poisoned = spectral_defense_filter(image, clf, threshold)
    if is_poisoned:
        return "Adversarial Image Detected"
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = fiba_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return f"Predicted Class: {predicted_class.item()}"

def visualize_result(image, result):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.set_title(result, fontsize=16, color='red')
    ax.axis('off')
    st.pyplot(fig)

# Load Models
@st.cache_resource
def load_models():
    spectral_model_path = 'spectral_defense_model.pkl'
    fiba_model = models.resnet50(pretrained=False)
    fiba_model.fc = torch.nn.Linear(fiba_model.fc.in_features, 8)
    checkpoint_path = 'checkpoints/ISIC2019/all2onedemo/best_acc_bd_ckpt.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    fiba_model.load_state_dict(checkpoint['netC'])
    fiba_model.eval()
    return spectral_model_path, fiba_model

# Streamlit Interface
st.title("SpectralDefense & FIBA Classification")
st.subheader("Detect adversarial images and classify benign images.")

spectral_model_path, fiba_model = load_models()

uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

threshold = st.slider("Detection Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    result = spectral_defense_pipeline(image, spectral_model_path, fiba_model, threshold)
    st.write(f"Result: **{result}**")
    visualize_result(image, result)

st.info("Upload an image to start the detection and classification process.")
