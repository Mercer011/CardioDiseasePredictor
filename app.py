import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- 1. Load the Model and Feature List ---
try:
    with open('cardio_model_pipeline.pkl', 'rb') as file:
        model_pipeline = pickle.load(file)
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Model or feature files not found. Ensure .pkl files are in the Colab root folder.")
    st.stop()


# --- 2. Define Prediction Function ---
def make_prediction(input_data):
    """Takes user input, converts it to a DataFrame, and returns the prediction."""
    # Create a DataFrame from the input, ensuring column order is correct
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Rationale: Calculate BMI on the raw user input, just like we did in the notebook.
    input_df['BMI'] = input_df['Weight_kg'] / (input_df['Height_cm'] / 100)**2
    
    # Use the pipeline for prediction. It automatically handles the scaling.
    prediction_proba = model_pipeline.predict_proba(input_df)[:, 1]
    prediction = model_pipeline.predict(input_df)[0]
    
    return prediction, prediction_proba[0]


# --- 3. Streamlit Interface ---

st.set_page_config(page_title="Cardiovascular Disease Predictor", layout="centered")

st.title("❤️ Cardiovascular Disease Risk Predictor")
st.markdown("Enter the patient's data below to get a real-time risk assessment.")

# --- Input Fields ---

col1, col2 = st.columns(2)

# Column 1 Inputs
with col1:
    age = st.slider("Age (Years)", 25, 70, 55)
    
    gender_map = {"Female (1)": 1, "Male (2)": 2}
    gender_input = st.selectbox("Gender", list(gender_map.keys()))
    gender = gender_map[gender_input]
    
    cholesterol = st.selectbox("Cholesterol Level", 
                               options=[1, 2, 3], 
                               format_func=lambda x: f"Level {x} (1: Normal, 3: High)")
    gluc = st.selectbox("Glucose Level", 
                        options=[1, 2, 3], 
                        format_func=lambda x: f"Level {x} (1: Normal, 3: High)")
    
# Column 2 Inputs
with col2:
    height = st.number_input("Height (cm)", 100, 200, 165)
    weight = st.number_input("Weight (kg)", 40.0, 150.0, 75.0)
    
    systolic_bp = st.number_input("Systolic BP (ap_hi)", 90, 240, 130)
    diastolic_bp = st.number_input("Diastolic BP (ap_lo)", 60, 180, 80)
    
    # Binary toggles (0 or 1)
    smokes = st.checkbox("Smokes", value=False)
    alco = st.checkbox("Drinks Alcohol", value=False)
    active = st.checkbox("Is Physically Active", value=True)

# --- Submission and Results ---

input_data = {
    'age': age,
    'Gender': gender,
    'Height_cm': height,
    'Weight_kg': weight,
    'Systolic_BP': systolic_bp,
    'Diastolic_BP': diastolic_bp,
    'Cholesterol_Level': cholesterol,
    'Glucose_Level': gluc,
    'Smokes': 1 if smokes else 0,
    'Drinks_Alcohol': 1 if alco else 0,
    'Is_Active': 1 if active else 0,
}

if st.button("Analyze Risk", type="primary"):
    # Perform input validation for BP
    if input_data['Systolic_BP'] <= input_data['Diastolic_BP']:
        st.error("⚠️ Input Error: Systolic BP must be greater than Diastolic BP.")
    else:
        # Make the prediction
        risk_class, risk_proba = make_prediction(input_data)
        risk_percent = risk_proba * 100
        
        st.subheader("Prediction Results:")

        if risk_class == 1:
            st.error(f"**High Risk Detected** 🔴")
            st.markdown(f"The model predicts a **{risk_percent:.1f}% probability** of having Cardiovascular Disease (CVH).")
        else:
            st.success(f"**Low Risk Detected** 🟢")
            st.markdown(f"The model predicts a **{risk_percent:.1f}% probability** of having Cardiovascular Disease (CVH).")

        st.markdown("---")
        # Displaying the calculated BMI
        calculated_bmi = input_data['Weight_kg'] / (input_data['Height_cm'] / 100)**2
        st.markdown(f"**Calculated BMI:** `{calculated_bmi:.2f}`")
