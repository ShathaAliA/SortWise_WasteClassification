import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import time


# 1. MONITORING & LOGGING CONFIGURATION

# Configures a local log file to automatically track application states.
logging.basicConfig(
    filename='sortwise_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

st.set_page_config(page_title="SortWise AI Portal", layout="centered")

# 2. ECO-FRIENDLY BACKGROUND & UI CUSTOMIZATION

st.markdown("""
    <style>
    /* Premium eco-cream background styling */
    .stApp {
        background-color: #F7F9F6; 
    }
    /* Main app forest green header design */
    h1 {
        color: #1B5E20 !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }
    .stMarkdown p {
        color: #37474F;
    }
    /* Style wrapper for the main primary prediction container */
    .prediction-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E0E4E0;
        margin-bottom: 15px;
    }
    /* Custom Styling for the Guidance Container Block */
    .guidance-box {
        background-color: #E8F5E9; 
        border-left: 6px solid #2E7D32; 
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .guidance-title {
        color: #1B5E20;
        font-weight: bold;
        font-size: 1.15rem;
        margin-bottom: 8px;
    }
    .guidance-text {
        color: #2E7D32;
        font-size: 1rem;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Main Application Headers
st.title("🌱 SortWise: AI Waste Classification Portal")
st.markdown("### Sustainable Household Recycling Support System")
st.write("Upload an image below to automatically classify your household item and get instant eco-disposal guidance.")

# System classes matching the target dataset architecture
CLASS_NAMES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']


# 3. DETAILED ECO-DISPOSAL GUIDANCE ENGINE

# Dictionary mapping predictions to explicit recycling instructions
GUIDANCE_MAPPING = {
    'Cardboard': "Flatten the box or container completely to conserve storage space. Place it directly into the Blue Paper/Cardboard Recycling Bin. Ensure it remains dry and completely free of organic food grease or liquids.",
    'Glass': "Carefully rinse out any organic residues or liquid contents. Remove separate caps or metal lids. Safely place the container into the designated Glass Collection Container for secondary melting processing.",
    'Metal': "Rinse thoroughly to remove any remaining food particles. Compress aluminum beverage cans or tin products if possible, then deposit them directly into your neighborhood's Metal Recycling Bin.",
    'Paper': "Keep all papers flat and entirely dry. Newspapers, notebooks, magazines, and office envelopes should go directly into the Blue Recycling Bin. Avoid recycling paper contaminated by oils or grease.",
    'Plastic': "Empty the container entirely and wash off internal residue. Crush the container tightly to decrease volume footprint, then secure it inside the Plastic Recycling Bin.",
    'Trash': "This material is composed of non-recyclable compounds or food-contaminated residuals. Please wrap it safely and discard it directly into the standard General Waste/Landfill Trash Bin to prevent contaminating recycling streams."
}
# 4. RUNTIME SYSTEM CACHING & ERROR HANDLING

@st.cache_resource
def load_sortwise_model():
    try:
        logging.info("Initializing local model load...")
        model = tf.keras.models.load_model('sortwise_model.h5') 
        logging.info("Model loaded successfully into RAM pipeline.")
        return model
    except Exception as e:
        logging.error(f"Critical Error: Failed to load model file: {str(e)}")
        st.error("System Failure: The model file 'sortwise_model.h5' is missing or corrupted. Check logs.")
        return None

model = load_sortwise_model()

# 5. USER ACTIONS AND PIPELINE EXECUTION

# Robust session state tracker to force cache clearing upon reset trigger
if 'uploader_id' not in st.session_state:
    st.session_state['uploader_id'] = 0

uploaded_file = st.file_uploader(
    "Choose a clear file from your device...", 
    type=["jpg", "jpeg", "png"], 
    key=f"file_picker_{st.session_state['uploader_id']}"
)

if uploaded_file is not None and model is not None:
    try:
        logging.info(f"User uploaded file for classification: {uploaded_file.name}")
        
        # Opens image securely inside RAM buffer (Transient Processing)
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Target Waste Object', use_container_width=True)
        
        with st.spinner('Running computer vision processing framework...'):
            start_time = time.time()
            
            # Standardized preprocessing normalization framework
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0  
            img_tensor = np.expand_dims(img_array, axis=0) 
            
            # Executing deep learning inference layers
            predictions = model.predict(img_tensor)
            
            # Sorting probabilities matrix to isolate secondary and alternative scores
            sorted_indices = np.argsort(predictions[0])[::-1]
            
            # 1st Highest Prediction (Primary Result)
            predicted_class = CLASS_NAMES[sorted_indices[0]]
            confidence_score = predictions[0][sorted_indices[0]] * 100
            
            # 2nd and 3rd Highest Predictions (Alternative Probabilities for Explainability)
            alt1_class = CLASS_NAMES[sorted_indices[1]]
            alt1_score = predictions[0][sorted_indices[1]] * 100
            
            alt2_class = CLASS_NAMES[sorted_indices[2]]
            alt2_score = predictions[0][sorted_indices[2]] * 100
            
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            
      
        # OUTPUT DISPLAY LAYER
     
        # Displaying the primary localized prediction card HTML wrapper
        st.markdown(f"""
            <div class="prediction-card">
                <h3 style='color: #2E7D32; margin-top:0;'>🟢 Primary Classification: {predicted_class}</h3>
                <p style='font-size: 1.1rem; margin-bottom:5px;'>Confidence Level: <b>{confidence_score:.2f}%</b></p>
                <p style='color: #757575; font-size: 0.9rem;'>System Latency: {inference_time_ms:.1f} ms</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Displaying Alternative Probabilities Section with native progress indicators
        st.markdown("#### 🔍 Alternative Model Probabilities (Explainability Metric)")
        st.write(f"1. {alt1_class}: {alt1_score:.2f}%")
        st.progress(int(alt1_score))
        
        st.write(f"2. {alt2_class}: {alt2_score:.2f}%")
        st.progress(int(alt2_score))
        
        # Fetching and displaying the designated recycling instructions container
        target_guidance = GUIDANCE_MAPPING.get(predicted_class, "No guidance guidelines available.")
        
        st.markdown(f"""
            <div class="guidance-box">
                            <div class="guidance-title">📋 Eco Disposal & Recycling Guidance</div>
                <div class="guidance-text">{target_guidance}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Logging classification tracking parameters
        logging.info(f"Inference Success: Pred={predicted_class}({confidence_score:.2f}%) | Alt1={alt1_class}({alt1_score:.2f}%) | Latency={inference_time_ms:.1f}ms")
        
        # "CLASSIFY ANOTHER" RESET BUTTON MECHANISM
        st.markdown("---")
        st.write("Do you want to process a completely different household item?")
        if st.button("🔄 Classify Another Item", use_container_width=True):
            # Increments uploader instance id string to force reset the cached input block states
            st.session_state['uploader_id'] += 1
            st.rerun()
            
    except Exception as e:
        logging.error(f"Runtime Exception while processing file {uploaded_file.name}: {str(e)}")
        st.error("File Error: Unable to extract data matrix from file. Ensure you upload a valid image.")
              
