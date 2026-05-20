import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load('xgboost_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# =========================
# TITLE
# =========================
st.title("📊 Customer Churn Prediction")
st.markdown("""
Predict whether a customer is likely to churn based on customer service and subscription information.
""")

st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("📌 About Project")

st.sidebar.info("""
This application predicts customer churn using an XGBoost Machine Learning model.

### Features Used:
- Tenure
- Monthly Charges
- Contract Type
- Internet Service
- Tech Support
- Payment Method
- Online Security
- Streaming Service
""")

st.sidebar.success("Model: XGBoost Classifier")

# =========================
# INPUT SECTION
# =========================
st.subheader("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# =========================
# PREDICTION BUTTON
# =========================
st.divider()

if st.button("🔍 Predict Churn"):

    # =========================
    # CREATE INPUT DATA
    # =========================
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,

        'Contract_One year': 0,
        'Contract_Two year': 0,

        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,

        'TechSupport_Yes': 0,
        'OnlineSecurity_Yes': 0,
        'StreamingMovies_Yes': 0,

        'PaymentMethod_Electronic check': 0
    }

    # =========================
    # CONTRACT
    # =========================
    if contract == "One year":
        input_data['Contract_One year'] = 1

    elif contract == "Two year":
        input_data['Contract_Two year'] = 1

    # =========================
    # INTERNET SERVICE
    # =========================
    if internet_service == "Fiber optic":
        input_data['InternetService_Fiber optic'] = 1

    elif internet_service == "No":
        input_data['InternetService_No'] = 1

    # =========================
    # TECH SUPPORT
    # =========================
    if tech_support == "Yes":
        input_data['TechSupport_Yes'] = 1

    # =========================
    # ONLINE SECURITY
    # =========================
    if online_security == "Yes":
        input_data['OnlineSecurity_Yes'] = 1

    # =========================
    # STREAMING MOVIES
    # =========================
    if streaming_movies == "Yes":
        input_data['StreamingMovies_Yes'] = 1

    # =========================
    # PAYMENT METHOD
    # =========================
    if payment_method == "Electronic check":
        input_data['PaymentMethod_Electronic check'] = 1

    # =========================
    # DATAFRAME
    # =========================
    input_df = pd.DataFrame([input_data])

    # =========================
    # MATCH MODEL COLUMNS
    # =========================
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # =========================
    # SCALING
    # =========================
    input_scaled = scaler.transform(input_df)

    # =========================
    # PREDICTION
    # =========================
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    # =========================
    # RESULT
    # =========================
    st.divider()

    st.subheader("📈 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Customer is likely to churn")

        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

        st.warning("""
        ### Business Recommendation
        - Offer retention promotion
        - Provide long-term contract discounts
        - Improve customer support quality
        """)

    else:

        st.success("✅ Customer is likely to stay")

        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

        st.info("""
        ### Business Insight
        Customer has a relatively low churn risk.
        Continue maintaining service quality and customer engagement.
        """)

# =========================
# FOOTER
# =========================
st.divider()

st.caption("""
Developed using Streamlit, XGBoost, and Docker.
""")