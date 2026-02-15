import streamlit as st 
import cv2
import numpy as np
import joblib

model=joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Brain tumor detection")

uploaded_file = st.file_uploader("Upload MRI Image",type=["jpg","png","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", width=300)
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)

    flat = scaler.transform(flat)
    prediction = model.predict(flat)
    
    if prediction[0]==1:
        st.error("Tumor detected")
    else:
        st.error("No abnormalities detected")
