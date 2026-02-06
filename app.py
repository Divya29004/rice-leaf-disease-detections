import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("rice_leaf_model.keras")

class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image to detect disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = np.max(prediction)*100
    predicted_class = class_names[np.argmax(prediction)]

    if confidence < 60:
        st.warning("Model is unsure. Possibly unknown disease.")
    else:
        st.success(f"Disease Detected: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")
