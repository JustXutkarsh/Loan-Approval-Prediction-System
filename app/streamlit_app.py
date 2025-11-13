import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1️⃣ Load trained model + encoders
# -------------------------------
model = joblib.load("loan_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant's details below to predict loan approval status.")

# -------------------------------
# 2️⃣ Input form for user
# -------------------------------
with st.form("loan_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0)
    loan_amount = st.number_input("Loan Amount (₹ in thousands)", min_value=0)
    loan_amount_term = st.selectbox("Loan Amount Term (in days)", [360, 180, 120, 480, 300])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Loan Approval")

# -------------------------------
# 3️⃣ Process input & Predict
# -------------------------------
if submitted:
    # Create DataFrame for model input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # Apply same label encodings used during training
    for col, le in encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])

    # Align columns with model training
    expected_features = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # -------------------------------
    # 4️⃣ Display result
    # -------------------------------
    if prediction == 1:
        st.success(f"✅ Loan Approved with {round(prediction_proba[1]*100, 2)}% confidence!")
    else:
        st.error(f"❌ Loan Not Approved with {round(prediction_proba[0]*100, 2)}% confidence!")

    st.write("---")
    st.subheader("📊 Input Summary")
    st.dataframe(input_data)

# -------------------------------
# 5️⃣ Helpful footer
# -------------------------------
st.markdown("""
---
🔍 *Model used:* RandomForestClassifier  
📦 *Trained using balanced dataset and saved encoders*  
💡 *Tip:* Try changing income, credit history, or property area to see how results vary.
""")
