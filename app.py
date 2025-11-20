import streamlit as st
import tensorflow as tf
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="GlioNet AI", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0E1117; color: #FAFAFA;}
    .stButton>button {width: 100%; border-radius: 20px; background-color: #FF4B4B; color: white;}
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 GlioNet: Volumetric Tumor Analysis")
st.markdown("### Clinical-Grade Glioblastoma Segmentation")
st.info("System Status: GOD MODE (Active) | Volume Calc: ENABLED")

with st.sidebar:
    st.header("Patient Data Upload")
    t1_file = st.file_uploader("Upload T1 Scan", type=['nii', 'nii.gz'])
    t1ce_file = st.file_uploader("Upload T1-CE (Contrast)", type=['nii', 'nii.gz'])
    t2_file = st.file_uploader("Upload T2 Scan", type=['nii', 'nii.gz'])
    flair_file = st.file_uploader("Upload FLAIR Scan", type=['nii', 'nii.gz'])
    st.markdown("---")
    st.caption("GlioNet 1.0 | Built by RJ")

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

    model = tf.keras.models.load_model(
        'NeuroVision_Pro_Dice_Final.keras', 
        custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef},
        compile=False
    )
    return model

def read_nifti_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    img = nib.load(tmp_path).get_fdata()
    os.remove(tmp_path)
    return img

# --- MAIN LOGIC ---
if t1_file and t1ce_file and t2_file and flair_file:
    
    try:
        # Load Volumes
        vol_t1 = read_nifti_file(t1_file)
        vol_t1ce = read_nifti_file(t1ce_file)
        vol_t2 = read_nifti_file(t2_file)
        vol_flair = read_nifti_file(flair_file)
        
        # Slider
        max_slices = vol_t1.shape[2]
        slice_index = st.slider("Select Slice for Visual Check", 0, max_slices-1, 90)
        
        # PREPARE SINGLE SLICE (For Display)
        raw_stack = np.stack([
            vol_t1[:, :, slice_index],
            vol_t1ce[:, :, slice_index],
            vol_t2[:, :, slice_index],
            vol_flair[:, :, slice_index]
        ], axis=-1)
        
        crop_start, crop_end = 56, 184
        img_cropped = raw_stack[crop_start:crop_end, crop_start:crop_end, :]
        img_norm = img_cropped / np.max(img_cropped)
        model_input = np.expand_dims(img_norm, axis=0)
        
        # 1. VISUAL DIAGNOSIS
        if st.button("Run Visual Diagnosis"):
            model = load_model()
            with st.spinner("Analyzing Slice..."):
                prediction = model.predict(model_input)
                pred_mask = np.argmax(prediction, axis=3)[0,:,:]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Input (T1-CE)")
                    st.image(img_norm[:,:,1], clamp=True, caption="Tumor Core Input")
                with col2:
                    st.subheader("Prediction")
                    fig, ax = plt.subplots()
                    ax.imshow(img_norm[:,:,3], cmap='gray')
                    ax.imshow(pred_mask, cmap='jet', alpha=0.5, vmin=0, vmax=3)
                    ax.axis('off')
                    st.pyplot(fig)
                with col3:
                    st.subheader("Analysis")
                    if np.max(pred_mask) > 0:
                        st.error("Tumor Detected")
                    else:
                        st.success("Clear")

        # 2. VOLUMETRIC CALCULATION
        st.markdown("---")
        st.header("3D Volumetric Analysis")
        if st.button("Calculate Total Tumor Volume"):
            model = load_model()
            total_tumor_pixels = 0
            
            # Progress Bar
            my_bar = st.progress(0)
            status_text = st.empty()
            
            # Loop through ALL slices
            for i in range(max_slices):
                # Update Progress
                progress = int((i / max_slices) * 100)
                my_bar.progress(progress)
                status_text.text(f"Scanning Slice {i}/{max_slices}...")
                
                # Prep Data (Same as single slice)
                s_stack = np.stack([
                    vol_t1[:, :, i], vol_t1ce[:, :, i], vol_t2[:, :, i], vol_flair[:, :, i]
                ], axis=-1)
                s_crop = s_stack[crop_start:crop_end, crop_start:crop_end, :]
                
                if np.max(s_crop) > 0:
                    s_norm = s_crop / np.max(s_crop)
                else:
                    s_norm = s_crop
                    
                s_input = np.expand_dims(s_norm, axis=0)
                
                # Predict (Fast Mode)
                p = model.predict(s_input, verbose=0)
                m = np.argmax(p, axis=3)[0,:,:]
                
                # Count Pixels
                total_tumor_pixels += np.sum(m > 0)
            
            # Complete
            my_bar.progress(100)
            status_text.text("Analysis Complete.")
            
            # Convert to cc (1 voxel = 1 mm^3 = 0.001 cc)
            volume_cc = total_tumor_pixels / 1000.0
            
            st.markdown(f"### 📊 Total Tumor Volume: **{volume_cc:.2f} cc**")
            
            if volume_cc > 100:
                st.error("Status: CRITICAL (Large Mass)")
            elif volume_cc > 10:
                st.warning("Status: OPERABLE (Medium Mass)")
            elif volume_cc > 0:
                st.info("Status: DETECTED (Small/Early Stage)")
            else:
                st.success("Status: CLEAN")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload all 4 files.")



