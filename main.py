import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import cv2

# Page configuration
st.set_page_config(
    page_title="Apple Leaf Disease Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }
    .result-card-healthy {
        background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #2E7D32;
    }
    .result-card-disease {
        background: linear-gradient(135deg, #ffebee 0%, #ef9a9a 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #c62828;
    }
    .result-card-warning {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCTIONS
# ============================================================

# FIXED CONFIDENCE THRESHOLD - DO NOT CHANGE
CONFIDENCE_THRESHOLD = 0.30  # 70% - Only accept predictions above this

def predict_with_confidence(test_image, model):
    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        
        predictions = model.predict(input_arr, verbose=0)
        
        # DEBUG: Print all predictions
        st.write("### Debug: All Predictions")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
        
        confidence = float(np.max(predictions) * 100)
        result_index = int(np.argmax(predictions))
        
        st.write(f"### Debug: Highest Prediction")
        st.write(f"Class: {class_names[result_index]}, Confidence: {confidence:.2f}%")
        
        if confidence >= CONFIDENCE_THRESHOLD * 100:
            return result_index, confidence, True
        else:
            return result_index, confidence, False
            
    except Exception as e:
        st.error(f"Error predicting disease: {e}")
        return None, None, False

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('trained_plant_disease_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================================
# LOAD MODEL
# ============================================================

model = load_model()

def is_apple_leaf_image(image):
    """
    Check if the image contains an apple leaf using multiple heuristics
    Returns: (is_leaf, reason)
    """
    try:
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # 1. Check image size
        if img_array.shape[0] < 50 or img_array.shape[1] < 50:
            return False, "Image is too small"
        
        # 2. Convert to HSV for color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # 3. Check for green color (apple leaves are green)
        # Green range in HSV: Hue 35-85
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.sum(green_mask) / (img_array.shape[0] * img_array.shape[1])
        
        # 4. Check for green color (apple leaves are green)
        if green_percentage < 0.05:  # Less than 5% green
            return False, "No green leaf detected in the image"
        
        # 5. Check for typical leaf shape (edge detection)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_percentage = np.sum(edges > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # A leaf should have some edges but not too many (noisy image)
        if edge_percentage < 0.01:
            return False, "No clear leaf structure detected"
        
        # 6. Check for blurriness (leaf should be somewhat sharp)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 30:
            return False, "Image is too blurry"
        
        # 7. Check if image is too uniform (solid color)
        std_brightness = np.std(gray)
        if std_brightness < 20:
            return False, "Image appears to be a solid color"
        
        return True, "Valid apple leaf image detected"
        
    except Exception as e:
        return False, f"Error processing image: {e}"

# ============================================================
# CLASS NAMES AND DISEASE INFO
# ============================================================

class_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

disease_info = {
    'Apple Scab': {
        'description': 'Apple scab is a common disease caused by the fungus Venturia inaequalis.',
        'treatment': 'Apply fungicides, remove fallen leaves, and prune infected branches.',
        'severity': 'Moderate to High'
    },
    'Black Rot': {
        'description': 'Black rot is caused by the fungus Botryosphaeria obtusa.',
        'treatment': 'Remove infected fruit and branches, apply fungicides, ensure proper sanitation.',
        'severity': 'High'
    },
    'Cedar Apple Rust': {
        'description': 'Cedar-apple rust is caused by the fungus Gymnosporangium juniperi-virginianae.',
        'treatment': 'Remove cedar trees nearby, apply fungicides, plant resistant varieties.',
        'severity': 'Moderate'
    },
    'Healthy': {
        'description': 'Your apple leaf appears to be healthy with no signs of disease.',
        'treatment': 'Continue regular maintenance and monitoring.',
        'severity': 'None'
    }
}

# ============================================================
# SIDEBAR
# ============================================================

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4134/4134160.png", width=80)
    st.title("🍎 Leaf Dashboard")
    st.markdown("---")
    
    app_mode = st.radio(
        "📌 Menu",
        ["🏠 Home", "ℹ️ About", "🔍 Predict Disease"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Made with ❤️ for farmers and agriculture enthusiasts")

# ============================================================
# HOME PAGE
# ============================================================

if app_mode == "🏠 Home":
    st.markdown('<h1 class="main-header">🍎 Apple Leaf Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Agriculture Solution</p>', unsafe_allow_html=True)
    
    try:
        st.image("home.jpg", caption='🍎 Apple Leaf Disease Detection', use_container_width=True)
    except:
        st.info("ℹ️ Please add a 'home.jpg' image to display here")
    
    st.markdown("""
    ### 🌿 Welcome to the Apple Leaf Disease Recognition System!
    
    Apples are one of the most widely grown and economically vital fruits in the world. 
    Apple farming supports millions of rural livelihoods, drives local economies, and secures global food supply chains. 
    However, orchards face constant threats from changing weather patterns, pests, and devastating diseases. 
    Early identification of these threats is critical to saving crops and maximizing yields. 
    
    This system bridges agricultural tradition with modern artificial intelligence. 
    By quickly analyzing images of your apple leaves, we help you detect problems before they spread across your entire orchard.
    ### 📋 How It Works
    
    1. **Upload Image:** Go to the **Disease Recognition** page
    2. **Analysis:** Our AI analyzes the leaf for diseases  
    3. **Results:** Get instant diagnosis and treatment recommendations
    
    ### 🚀 Get Started
    
    Click on **🔍 Predict Disease** in the sidebar to begin!
    """)
    
    st.info("💡 **Tip:** For best results, upload clear, well-lit images of the entire leaf.")

# ============================================================
# ABOUT PAGE
# ============================================================

elif app_mode == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About Apple Leaf Diseases</h1>', unsafe_allow_html=True)
    
    # ============================================================
    # SECTION 1: Apple Leaf Disease Overview
    # ============================================================
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5faf5 0%, #e8f5e9 100%); padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
        <h2 style="color: #1B5E20; text-align: center;">🍎 Common Apple Leaf Diseases</h2>
        <p style="text-align: center; color: #444; font-size: 1.1rem;">
            Apple trees are susceptible to various diseases that affect their leaves, fruits, and overall health.
            Early detection is key to preventing crop loss.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # Disease 1: Healthy Leaf (FIRST)
    # ============================================================
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            st.image("Healthy1 (1).jpg", caption="✅ Healthy Leaf", use_container_width=True)
        except:
            st.image("https://cdn.shopify.com/s/files/1/0058/7954/4885/files/healthy_apple_leaf_1024x1024.jpg", caption="✅ Healthy Leaf", use_container_width=True)
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); height: 100%; border-top: 4px solid #2E7D32;">
            <h4 style="color: #1B5E20;">✅ Healthy Leaf</h4>
            <p style="color: #666; font-size: 0.95rem;">
                <strong>Signs:</strong> Clean, vibrant green color<br>
                <strong>Characteristics:</strong> No spots, lesions, or discoloration<br>
                <strong>Status:</strong> Tree is thriving<br>
                <strong>Action:</strong> Continue regular maintenance and monitoring
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # Disease 2: Apple Scab
    # ============================================================
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            st.image("applescab (1).jpg", caption="🍂 Apple Scab", use_container_width=True)
        except:
            st.image("https://plantdiseasehandbook.tamu.edu/files/2020/11/Apple-scab-leaves.jpg", caption="🍂 Apple Scab", use_container_width=True)
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); height: 100%; border-top: 4px solid #795548;">
            <h4 style="color: #4E342E;">🍂 Apple Scab</h4>
            <p style="color: #666; font-size: 0.95rem;">
                <strong>Caused by:</strong> Fungus <em>Venturia inaequalis</em><br>
                <strong>Signs:</strong> Olive-green to brown velvety spots on leaves<br>
                <strong>Impact:</strong> Leaves may drop prematurely<br>
                <strong>Treatment:</strong> Use targeted fungicides containing active ingredients like myclobutanil or captan during rainy periods, remove fallen leaves
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # Disease 3: Black Rot
    # ============================================================
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            st.image("appleblackdot (4).jpg", caption="🍂 Black Rot", use_container_width=True)
        except:
            st.image("https://extension.umd.edu/sites/default/files/styles/scale_crop_1200x630/public/2021-10/Black%20rot%20apple%20leaf.jpg", caption="🍂 Black Rot", use_container_width=True)
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); height: 100%; border-top: 4px solid #D32F2F;">
            <h4 style="color: #B71C1C;">🍂 Black Rot</h4>
            <p style="color: #666; font-size: 0.95rem;">
                <strong>Caused by:</strong> Fungus <em>Botryosphaeria obtusa</em><br>
                <strong>Signs:</strong> Dark brown to black lesions on leaves<br>
                <strong>Impact:</strong> Can kill branches and reduce yield<br>
                <strong>Treatment:</strong> Remove infected branches, Apply fungicides like captan or thiophanate-methyl from pink bud stage through harvest.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # Disease 4: Cedar Apple Rust
    # ============================================================
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            st.image("appleCiderRust.jpg.JPG", caption="🟠 Cedar Apple Rust", use_container_width=True)
        except:
            st.image("https://extension.umd.edu/sites/default/files/styles/scale_crop_1200x630/public/2021-10/Cedar%20Apple%20Rust%20leaf.jpg", caption="🟠 Cedar Apple Rust", use_container_width=True)
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); height: 100%; border-top: 4px solid #EF6C00;">
            <h4 style="color: #E65100;">🟠 Cedar Apple Rust</h4>
            <p style="color: #666; font-size: 0.95rem;">
                <strong>Caused by:</strong> Fungus <em>Gymnosporangium juniperi-virginianae</em><br>
                <strong>Signs:</strong> Yellow-orange spots with tiny black dots<br>
                <strong>Impact:</strong> Can cause defoliation<br>
                <strong>Treatment:</strong> Remove cedar trees nearby, Spray protective fungicides like myclobutanil or mancozeb from the tight cluster stage until petal fall
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem; margin: 1rem 0;">
        <p style="color: #888; font-size: 0.9rem;">
            📸 <strong>Tip:</strong> For best results, upload clear images showing the entire leaf with good lighting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # SECTION 2: Project Information
    # ============================================================
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin: 2rem 0;">
        <h2 style="color: #1B5E20;">📊 About This System</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📂 Dataset Information
        
        This dataset is recreated using offline augmentation from the original dataset available on Kaggle.
        
        **Dataset Statistics:**
        - 📚 **Training:** 658 images
        - ✅ **Validation:** 291 images
        - 🎯 **Classes:** 4 disease categories
        
        **Disease Classes:**
        - ✅ Healthy
        - 🍂 Apple Scab
        - 🍂 Black Rot
        - 🍂 Cedar Apple Rust
        
        [🔗 View Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Technology Stack
        
        | Component | Technology |
        |-----------|------------|
        | **Framework** | TensorFlow / Keras |
        | **Frontend** | Streamlit |
        | **Model** | CNN (Convolutional Neural Network) |
        | **Image Processing** | PIL, OpenCV, NumPy |
        | **Data Analysis** | Pandas, Matplotlib |
        
        ### 🎯 Project Goals
        
        1. Provide accurate disease detection for farmers
        2. Enable early intervention to reduce crop loss
        3. Make agricultural diagnostics accessible
        4. Promote sustainable farming practices
        """)
    
    # ============================================================
    # SECTION 3: Why It Matters
    # ============================================================
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 2rem; border-radius: 16px; margin: 2rem 0;">
        <h2 style="color: #1B5E20;">👨‍🌾 Why This Matters</h2>
        <p style="color: #333; font-size: 1.05rem; line-height: 1.8;">
            Apple leaf diseases can significantly impact crop yield and quality. 
            Early detection and proper treatment are crucial for maintaining healthy orchards. 
            Our AI-powered system helps farmers and gardeners quickly identify diseases and take appropriate action.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("🌱 **Together, we can build a more sustainable future for agriculture!**")
    
    
# ============================================================
# PREDICT DISEASE PAGE
# ============================================================

elif app_mode == "🔍 Predict Disease":
    st.markdown('<h1 class="main-header">🔍 Disease Recognition</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        test_image = st.file_uploader(
            "Choose an image of an apple leaf",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if test_image is not None:
            st.markdown("---")
            
            if st.button("🔍 Predict Disease", use_container_width=True):
                if model is None:
                    st.error("❌ Model not loaded. Please check the model file.")
                else:
                    with st.spinner("Analyzing the leaf... Please wait..."):
                        time.sleep(0.5)
                        
                        try:
                            # Open image for validation
                            img = Image.open(test_image)
                            
                            # STEP 1: Check if this is actually an apple leaf
                            is_leaf, leaf_message = is_apple_leaf_image(img)
                            
                            if not is_leaf:
                                st.markdown(f"""
                                <div style="background: #fff3e0; padding: 1.5rem; border-radius: 12px; border-left: 5px solid #ff9800; margin: 1rem 0;">
                                    <h3 style="color: #e65100;">❌ Invalid Image</h3>
                                    <p><strong>Issue:</strong> {leaf_message}</p>
                                    <p>Please upload a clear image of an apple leaf.</p>
                                    <p><strong>Requirements:</strong></p>
                                    <ul>
                                        <li>📸 Clear, well-lit image</li>
                                        <li>🍃 Should show the entire leaf</li>
                                        <li>🌿 Green color should be visible</li>
                                        <li>📐 Leaf should fill most of the frame</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                                st.stop()
                            
                            # STEP 2: If valid, make prediction
                            result_index, confidence, is_valid = predict_with_confidence(test_image, model)

                            if result_index is None:
                                st.error("Error in prediction")
                            else:
                                disease_name = class_names[result_index]
                                is_healthy = disease_name == "Healthy"
                                
                                if not is_valid:
                                    # Low confidence - even though it's a leaf, it's unclear
                                    st.markdown(f"""
                                    <div style="background: #fff3e0; padding: 1.5rem; border-radius: 12px; border-left: 5px solid #ff9800; margin: 1rem 0;">
                                        <h3 style="color: #e65100;">⚠️ Unclear Image</h3>
                                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                        <p><strong>Minimum Required:</strong> 70%</p>
                                        <p>The image is too blurry or unclear for a confident diagnosis.</p>
                                        <p><strong>Best guess:</strong> {disease_name}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Valid prediction with good confidence
                                    card_class = "result-card-healthy" if is_healthy else "result-card-disease"
                                    st.markdown(f"""
                                    <div class="{card_class}">
                                        <h3>{'✅' if is_healthy else '⚠️'} {disease_name}</h3>
                                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                                        <p><strong>Severity:</strong> {disease_info[disease_name]['severity']}</p>
                                        <hr>
                                        <p><strong>Description:</strong> {disease_info[disease_name]['description']}</p>
                                        <p><strong>Recommended Treatment:</strong> {disease_info[disease_name]['treatment']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if not is_healthy:
                                        st.warning("⚠️ **Action Required:** Consult with an agricultural expert for proper treatment implementation.")
                                    else:
                                        st.success("🌿 **Great news!** Your plant appears healthy. Continue regular care and monitoring.")
                                    
                                    # Display confidence bar
                                    st.markdown("### 📈 Confidence Level")
                                    st.progress(float(confidence/100))
                                    
                        except Exception as e:
                            st.error(f"Error processing image: {e}")
    
    with col2:
        if test_image is not None:
            st.markdown("### 🖼️ Uploaded Image")
            try:
                image = Image.open(test_image)
                st.image(image, caption='Apple Leaf Sample', use_container_width=True)
            except:
                st.error("Error loading image. Please try uploading again.")
            
            st.markdown("---")
            st.markdown("### 📋 Tips for Best Results")
            st.info("""
            - 📸 Use clear, well-lit images
            - 🍃 Capture the entire leaf
            - 🎯 Ensure the leaf fills most of the frame
            - 🌿 Take photos from straight above
            - ❌ Avoid blurry or noisy images
            """)
        else:
            st.markdown("### 📸 No Image Uploaded")
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background-color: #f5f5f5; border-radius: 10px;">
                <div style="font-size: 5rem;">📤</div>
                <h3>Upload an image to begin</h3>
                <p>Click the "Browse files" button to select an image</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🔬 Powered by Deep Learning | 🌱 For healthier apple crops")