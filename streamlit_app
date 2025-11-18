import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Load Dataset + Model
# ---------------------------

DATA_PATH = "beneficiary_credit_20k.csv"
MODEL_PATH = "ann_credit_score.h5"

st.set_page_config(page_title="Credit Score Predictor", layout="centered")
st.title("🔮 Beneficiary Credit Score Predictor (Scaled Inputs)")
st.write("Model trained with StandardScaler — scaling is applied automatically.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Select same features used during training
selected_features = [
    'annual_income','joint_income','property_value','business_income',
    'loan_amount','repayment_score','beneficiary_count','age'
]

# Fit StandardScaler on training features
scaler = StandardScaler()
scaler.fit(df[selected_features])

# Load trained ANN model
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH, compile=False)

model = load_trained_model()

# ---------------------------
# Streamlit Inputs
# ---------------------------
st.subheader("Enter Applicant Details")

annual_income = st.number_input("Annual Income", min_value=0.0)
joint_income = st.number_input("Joint Income", min_value=0.0)
property_value = st.number_input("Property Value", min_value=0.0)
business_income = st.number_input("Business Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
repayment_score = st.number_input("Repayment Score", min_value=0.0)
beneficiary_count = st.number_input("Beneficiary Count", min_value=0)
age = st.number_input("Age", min_value=0)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict Credit Score"):

    # Step 1: Create input vector
    input_data = np.array([
        annual_income,
        joint_income,
        property_value,
        business_income,
        loan_amount,
        repayment_score,
        beneficiary_count,
        age
    ]).reshape(1, -1)

    # Step 2: Scale input using StandardScaler
    scaled_input = scaler.transform(input_data)

    # Step 3: Predict using model
    prediction = model.predict(scaled_input)
    predicted_score = float(prediction[0][0])

    # Step 4: Display result
    st.success(f"Predicted Credit Score: *{predicted_score:.2f}*")
