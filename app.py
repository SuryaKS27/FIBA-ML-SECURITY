import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from torchvision import transforms, models
import torch
import joblib
import os
import matplotlib.pyplot as plt

# Function Definitions
def calculate_fourier_spectrum(im):
    im = np.array(im.convert('L'))  # Convert to grayscale
    fft = np.fft.fft2(im)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
    return magnitude_spectrum

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

def visualize_fft_comparison(clean_image, attacked_image):
    clean_fft = calculate_fourier_spectrum(clean_image)
    attacked_fft = calculate_fourier_spectrum(attacked_image)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].imshow(clean_image)
    axes[0, 0].set_title("Clean Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(clean_fft, cmap='gray')
    axes[0, 1].set_title("FFT of Clean Image")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(attacked_image)
    axes[1, 0].set_title("Attacked Image")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(attacked_fft, cmap='gray')
    axes[1, 1].set_title("FFT of Attacked Image")
    axes[1, 1].axis('off')
    
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
st.set_page_config(page_title="SpectralDefense & FIBA Classification", layout="wide")

# Home Page
def home_page():
    # Layout for Home Page
    st.title("SpectralDefense & FIBA Classification")
    st.subheader("Detect adversarial images and classify benign images.")

    # Adding team names and some more context
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("OIP.jpeg", width=150)  # Placeholder image
    with col2:
        st.markdown(
            """
            ### Developed by:
            - **Surya K S**
            - **Ramya M N**
            
            ### Project Overview:
            The SpectralDefense system uses frequency domain analysis (FFT) to detect adversarial examples and classify images into benign or adversarial categories.
            """
        )
    st.markdown("___")
    st.info("Select an option from the sidebar to start exploring the app.")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = ["Home", "Image-Level Distribution", "Color Analysis", "Frequency Analysis (FFT)"]
choice = st.sidebar.radio("Go to", options)

# Load Models
spectral_model_path, fiba_model = load_models()

# Home Page
if choice == "Home":
    home_page()

# Image-Level Distribution
elif choice == "Image-Level Distribution":
    uploaded_csv = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_csv:
        data = pd.read_csv(uploaded_csv)
        st.sidebar.success("CSV File Loaded")

        st.title("Image-Level Distribution")
        st.subheader("Bar Chart")
        bar_fig = px.bar(data['label'].value_counts(), x=data['label'].unique(), y=data['label'].value_counts().values, labels={'x': 'Category', 'y': 'Count'})
        st.plotly_chart(bar_fig)

        st.subheader("Pie Chart")
        pie_fig = px.pie(data, names='label', title="Category Proportions")
        st.plotly_chart(pie_fig)

# Color Analysis Section
elif choice == "Color Analysis":
    st.title("Color Analysis")
    image_folder = st.sidebar.text_input("Enter Image Folder Path")
    if image_folder:
        available_images = os.listdir('ISIC_2019_Training_Input/ISIC_2019_Training_Input')[:10]
        images = [Image.open(os.path.join(image_folder, img)) for img in available_images]
    
    r_hist, g_hist, b_hist = [], [], []
    for img in images:
        image = np.array(img)
        r_hist.append(np.histogram(image[:, :, 0], bins=256, range=(0, 256))[0])
        g_hist.append(np.histogram(image[:, :, 1], bins=256, range=(0, 256))[0])
        b_hist.append(np.histogram(image[:, :, 2], bins=256, range=(0, 256))[0])

    avg_r = np.mean(r_hist, axis=0)
    avg_g = np.mean(g_hist, axis=0)
    avg_b = np.mean(b_hist, axis=0)

    color_df = pd.DataFrame({'Red': avg_r, 'Green': avg_g, 'Blue': avg_b})
    color_fig = px.line(color_df, title="RGB Channel Intensities")
    st.plotly_chart(color_fig)

# Frequency Analysis (FFT)
elif choice == "Frequency Analysis (FFT)":
    st.title("Frequency Analysis")
    clean_image_file = st.file_uploader("Upload Clean Image for FFT", type=["png", "jpg", "jpeg"], key="clean")
    attacked_image_file = st.file_uploader("Upload Attacked Image for FFT", type=["png", "jpg", "jpeg"], key="attacked")
    
    if clean_image_file and attacked_image_file:
        clean_image = Image.open(clean_image_file)
        attacked_image = Image.open(attacked_image_file)
        visualize_fft_comparison(clean_image, attacked_image)

st.info("Upload an image to start the detection and classification process.")
