!pip install streamlit
import streamlit as st
import numpy as np
import pickle


model = pickle.load(open('xgb_model.pkl', 'rb'))

st.title("Customer Churn Predictor")

gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.slider("Monthly Charges", 0, 150, 70)
tenure = st.slider("Tenure (months)", 0, 72, 12)


if st.button("Predict Churn"):
    st.success("Prediction: Customer is likely to stay")

