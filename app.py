import zipfile
import os

# --- UNZIP THE MODEL IF NOT ALREADY DONE ---
if not os.path.exists("my_model"):
    with zipfile.ZipFile("my_model.zip", "r") as zip_ref:
        zip_ref.extractall("my_model")


import streamlit as st
from PIL import Image
import numpy as np
import keras

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bottle Cap Defect Detector",
    page_icon="🍺",
    layout="wide"
)

# --- SIDEBAR FOR PROJECT INFORMATION ---
st.sidebar.header("About This Project")
st.sidebar.info("""
This application uses a deep learning model to detect whether a bottle has a cap or not.
Bottles with caps are considered "Good", and bottles without caps are considered "Defective".
The model is a Convolutional Neural Network (CNN) trained using TensorFlow and Keras.
""")
st.sidebar.success("Project by: Jash")

# --- MODEL LOADING ---
@st.cache_resource
def load_keras_model():
    labels_path = "labels.txt"
    model_path = "my_model"

    # Load the labels
    with open(labels_path, "r") as f:
        labels = [line.strip().split(maxsplit=1)[1] for line in f if line.strip()]

    # Load the SavedModel as a Keras Layer
    model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    return model_layer, labels



    # Load the labels
    with open(labels_path, "r") as f:
        labels = [line.strip().split(maxsplit=1)[1] for line in f if line.strip()]


    # Load the SavedModel as a Keras Layer
    model_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    return model_layer, labels

# --- PREDICTION FUNCTION ---
def predict(image_to_predict, model_layer, labels):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = image_to_predict.resize(size)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction_output = model_layer(data)
    prediction_tensor = list(prediction_output.values())[0]
    prediction = prediction_tensor.numpy()[0]
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[index]
    return class_name, confidence_score

# --- MAIN APP INTERFACE ---
st.title("🍺 Bottle Cap Detection System")

with st.expander("ℹ How to Use This App"):
    st.write("""
    1. **Upload an image** of a bottle with or without a cap.
    2. The AI model will analyze the image and determine if it is Good (with cap) or Defective (without cap).
    3. The **Status** and **Confidence Score** will be displayed on the right.
    """)

uploaded_file = st.file_uploader("Upload a bottle image for inspection...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.header("Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Prediction")
        with st.spinner('Analyzing the image...'):
            model_layer, labels = load_keras_model()
            class_name, confidence_score = predict(image, model_layer, labels)

        if class_name.lower() == "defective":
            st.error(f"Status: {class_name}")
            st.write(f"**Confidence:** {confidence_score:.2%}")
            st.warning("**Recommendation:** This bottle should be flagged for inspection.")
        else:
            st.success(f"Status: {class_name}")
            st.write(f"**Confidence:** {confidence_score:.2%}")
            st.info("**Recommendation:** This bottle has passed the cap inspection.")

else:
    st.header("Example Cases")
    st.write("No image uploaded yet. Check out these examples of Good and Defective bottles:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Good (With Cap)")
        st.image("good.png", caption="A bottle with a cap.")
    
    with col2:
        st.subheader("Defective (No Cap)")
        st.image("defective.png", caption="A bottle without a cap.")


        
