from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_model.pkl"

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")
st.title("Heart Disease Prediction")
st.caption("Educational machine learning model for a class project. This is not a medical diagnosis.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest pain type", [0, 1, 2, 3], format_func=lambda x: {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}.get(x, str(x)))
    trestbps = st.number_input("Resting blood pressure", min_value=50, max_value=220, value=120)
    chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}.get(x, str(x)))
    thalach = st.number_input("Maximum heart rate", min_value=50, max_value=220, value=150)
    exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope", [0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}.get(x, str(x)))
    ca = st.number_input("Number of major vessels", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia result", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}.get(x, str(x)))

    submitted = st.form_submit_button("Predict")

if submitted:
    patient = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
    ])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    st.subheader("Prediction result")
    st.write(f"Model prediction: {'heart disease class detected' if prediction == 1 else 'no heart disease class detected'}")
    st.write(f"Probability of heart disease: {probability:.2%}")
    st.warning("This tool is for educational purposes only and is not a medical diagnosis.")
