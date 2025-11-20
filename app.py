import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="NeuroVision Pro", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    .stButton>button {width: 100%; border-radius: 20px; background-color: #FF4B4B; color: white;}
    .css-1v0mbdj {display: flex; justify-content: center;} 
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 NeuroVision Pro: Multi-Modal Glioblastoma Detection")
st.markdown("### Clinical-Grade AI (Dice Score: 0.9919)")
st.info("System Status: Active")

with st.sidebar:
    st.header("Patient Data Upload")
    st.write("Upload raw NIfTI files (.nii) to begin.")
    t1_file = st.file_uploader("Upload T1 Scan", type=['nii', 'nii.gz'])
    t1ce_file = st.file_uploader("Upload T1-CE (Contrast)", type=['nii', 'nii.gz'])
    t2_file = st.file_uploader("Upload T2 Scan", type=['nii', 'nii.gz'])
    flair_file = st.file_uploader("Upload FLAIR Scan", type=['nii', 'nii.gz'])
    st.markdown("---")
    st.caption("NeuroVision v3.0 | Built by Rijul")

# --- HELPER: LOAD MODEL ---
@st.cache_resource
def load_model():
    def dice_coef(y_true, y_pred, smooth=1):
        import tensorflow.keras.backend as K
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    # Load your FINAL GOD MODE model
    model = tf.keras.models.load_model(
        'NeuroVision_Pro_Dice_Final.keras', 
        custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef},
        compile=False
    )
    return model

# --- HELPER: READ RAW FILE ---
def read_nifti_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    img = nib.load(tmp_path).get_fdata()
    os.remove(tmp_path)
    return img

# --- MAIN EXECUTION ---
if t1_file and t1ce_file and t2_file and flair_file:
    
    try:
        # 1. Load Volumes
        vol_t1 = read_nifti_file(t1_file)
        vol_t1ce = read_nifti_file(t1ce_file)
        vol_t2 = read_nifti_file(t2_file)
        vol_flair = read_nifti_file(flair_file)
        
        # 2. Slider
        max_slices = vol_t1.shape[2]
        slice_index = st.slider("Select Brain Slice", 0, max_slices-1, 90)
        
        # 3. Stack Raw
        raw_stack = np.stack([
            vol_t1[:, :, slice_index],
            vol_t1ce[:, :, slice_index],
            vol_t2[:, :, slice_index],
            vol_flair[:, :, slice_index]
        ], axis=-1)
        
        # 4. Crop (The Magic Fix)
        crop_start = 56
        crop_end = 184
        img_cropped = raw_stack[crop_start:crop_end, crop_start:crop_end, :]
        
        # 5. Normalize
        img_norm = img_cropped / np.max(img_cropped)
        model_input = np.expand_dims(img_norm, axis=0)
        
        if st.button("Run Diagnosis"):
            model = load_model()
            
            with st.spinner("NeuroVision is Analyzing Tissues..."):
                prediction = model.predict(model_input)
                
                # Extract Data
                tumor_confidence = prediction[0, :, :, 3]
                pred_mask = np.argmax(prediction, axis=3)[0,:,:]
                max_conf = np.max(tumor_confidence)
                
                # --- LAYOUT ---
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Input (T1-CE)")
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(img_norm[:,:,1], cmap='gray') # Show T1-CE channel
                    ax1.axis('off')
                    st.pyplot(fig1)
                    
                with col2:
                    st.subheader("AI Heatmap")
                    fig2, ax2 = plt.subplots()
                    im = ax2.imshow(tumor_confidence, cmap='hot', vmin=0, vmax=1)
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    st.pyplot(fig2)
                    
                with col3:
                    st.subheader("Tumor Segmentation")
                    fig3, ax3 = plt.subplots()
                    ax3.imshow(img_norm[:,:,3], cmap='gray') # Background is FLAIR
                    
                    if max_conf > 0.01:
                        # Only show mask if confidence is high enough
                        ax3.imshow(pred_mask, cmap='jet', alpha=0.5, vmin=0, vmax=3)
                    else:
                        st.warning("No Tumor Found")
                        
                    ax3.axis('off')
                    st.pyplot(fig3)
                
                # --- CONCLUSION ---
                st.markdown("---")
                if max_conf > 0.90:
                    st.error(f"⚠️ CRITICAL DETECTION: High-Confidence Tumor Signal ({max_conf*100:.2f}%)")
                    
                elif max_conf > 0.50:
                    st.warning(f"⚠️ DETECTION: Potential Abnormalities Found ({max_conf*100:.2f}%)")
                else:
                    st.success(f"✅ CLEAR: No significant tumor markers detected ({max_conf*100:.2f}%)")

    except Exception as e:
        st.error(f"Error processing scan: {e}")

else:
    st.info("Upload all 4 MRI modalities to initialize the system.")