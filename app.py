import streamlit as st
import joblib
import numpy as np

model = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Smart Crop Recommender")
st.markdown("Enter soil and environmental conditions to get crop recommendations:")

col1, col2 = st.columns(2)

with col1:
    n = st.slider("Nitrogen (N) ppm", 0, 150, 50)
    p = st.slider("Phosphorus (P) ppm", 0, 150, 50)
    k = st.slider("Potassium (K) ppm", 0, 200, 50)

with col2:
    ph = st.slider("Soil pH", 3.0, 10.0, 6.5, 0.1)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 1.0)
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)

if st.button("Get Crop Recommendation"):

    input_data = np.array([[n, p, k, ph, humidity, temp]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    crop_name = le.inverse_transform(prediction)[0]
   
    st.subheader(f"Recommended Crop: {crop_name.capitalize()}")
    st.markdown("---")

    st.write("**Input Values:**")
    st.json({
        "Nitrogen (N)": f"{n} ppm",
        "Phosphorus (P)": f"{p} ppm",
        "Potassium (K)": f"{k} ppm",
        "Soil pH": ph,
        "Humidity": f"{humidity}%",
        "Temperature": f"{temp}°C"
    })
