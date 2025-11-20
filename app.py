import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE ---
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- TITLE & HEADER ---
st.title("🧠 NeuroVision: AI-Powered Tumor Detection")
st.markdown("### Automated Glioblastoma Segmentation System")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    st.write("Upload a FLAIR MRI scan (.nii file) to detect tumor regions.")
    uploaded_file = st.file_uploader("Choose a NIfTI file", type=['nii', 'nii.gz'])
    
    confidence_threshold = st.slider("Segmentation Confidence", 0.0, 1.0, 0.5)
    st.info("NeuroVision v1.0 | Built by Rijul")

# --- FUNCTIONS ---

@st.cache_resource
def load_model():
    # Load your trained baby
    model = tf.keras.models.load_model('NeuroVision_Baby_v1.keras', compile=False)
    return model

def preprocess_image(nifti_file):
    # 1. Save uploaded file temporarily so Nibabel can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
        tmp_file.write(nifti_file.read())
        tmp_path = tmp_file.name

    # 2. Load the NIfTI file
    img = nib.load(tmp_path).get_fdata()
    os.remove(tmp_path) # Clean up

    # 3. Pick the middle slice (Best view of the brain)
    total_slices = img.shape[2]
    slice_idx = total_slices // 2
    img_slice = img[:, :, slice_idx]

    # 4. Resize to 128x128 (What the Baby expects)
    img_resized = cv2.resize(img_slice, (128, 128))

    # 5. Normalize (0-1 range)
    img_resized = img_resized / np.max(img_resized)

    # 6. Prepare for Model (Batch Size 1, 128, 128, 4 Channels)
    # Since we only have 1 file, we stack it 4 times to fake the other channels
    img_input = np.stack([img_resized]*4, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

    return img_slice, img_resized, img_input

# --- MAIN EXECUTION ---

if uploaded_file is not None:
    st.write("Scanning Brain Structure...")
    
    # Load Model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure the .keras file is in the same folder!")
        st.stop()

    # Preprocess
    original_slice, display_slice, model_input = preprocess_image(uploaded_file)

    # Predict
    if st.button("Analyze MRI Scan"):
        with st.spinner('Running Neural Network...'):
            prediction = model.predict(model_input)
            
            # Process Prediction (Get the Tumor Mask)
            # We look for class 1, 2, or 3 (Tumor parts)
            pred_mask = np.argmax(prediction, axis=3)[0,:,:]
            
            # Create a layout
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Input Scan (FLAIR)")
                fig1, ax1 = plt.subplots()
                ax1.imshow(display_slice, cmap='gray')
                ax1.axis('off')
                st.pyplot(fig1)

            with col2:
                st.subheader("AI Segmentation Mask")
                fig2, ax2 = plt.subplots()
                # 'jet' makes the background blue and tumor red/yellow
                ax2.imshow(pred_mask, cmap='jet', vmin=0, vmax=3) 
                ax2.axis('off')
                st.pyplot(fig2)

            with col3:
                st.subheader("Overlay (Augmented)")
                fig3, ax3 = plt.subplots()
                ax3.imshow(display_slice, cmap='gray')
                # Overlay the mask with transparency
                ax3.imshow(pred_mask, cmap='jet', alpha=0.5, vmin=0, vmax=3)
                ax3.axis('off')
                st.pyplot(fig3)

            st.success("Tumor Detected Successfully.")

else:

    st.warning("Please upload an MRI scan to begin analysis.")
