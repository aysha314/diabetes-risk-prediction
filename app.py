import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("diabetes_model.pkl")

# Page settings
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# Title
st.title("🩺 Diabetes Risk Prediction")

st.markdown(
    "Enter patient health details below to assess diabetes risk."
)

st.divider()

# Input Form
with st.form("prediction_form"):

    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            value=100
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            value=20
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=0,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            value=30
        )

    predict_button = st.form_submit_button(
        "Predict Risk"
    )

# Prediction
if predict_button:

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.metric(
        "Probability Score",
        f"{probability * 100:.2f}%"
    )

    if probability < 0.30:
        st.success("Risk Level: Low")
    elif probability < 0.60:
        st.warning("Risk Level: Moderate")
    else:
        st.error("Risk Level: High")

    st.subheader("Recommendation")

    if prediction == 1:
        st.warning(
            """
            • Consult a healthcare professional

            • Monitor blood glucose levels regularly

            • Follow a balanced diet

            • Maintain regular physical activity
            """
        )
    else:
        st.info(
            """
            • Continue a healthy lifestyle

            • Exercise regularly

            • Maintain a balanced diet

            • Schedule routine health checkups
            """
        )

st.divider()

st.caption(
    "For educational purposes only. This application does not replace professional medical advice."
)