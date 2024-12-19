import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model, feature_names = joblib.load("adaboost_model.pkl")  # Ensure this file contains the model and feature names

# Streamlit app title
st.title("Bitcoin Price Prediction: High vs. Low")

# Introduction
st.write("""
This app predicts whether the Bitcoin price category is **High** or **Low** based on the provided features. 
Fill in the fields below and click **Predict** to get the results.
""")

# Input fields for features
st.header("Input Bitcoin Features")
open_price = st.number_input("Open Price (in USD)", value=0.0, step=0.01)
high_price = st.number_input("High Price (in USD)", value=0.0, step=0.01)
low_price = st.number_input("Low Price (in USD)", value=0.0, step=0.01)
volume = st.number_input("Volume (in millions)", value=0.0, step=1.0)

# Prepare the input data
input_data = pd.DataFrame({
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price],
    'Volume': [volume]
})

# Ensure the input data matches the model's feature names
try:
    input_data = input_data[feature_names]
except KeyError:
    st.error("Feature mismatch! Ensure the input fields match the model's expected features.")

# Display the input data for confirmation
st.write("Input Data Preview:")
st.write(input_data)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        result = "High" if prediction[0] == 1 else "Low"
        st.success(f"The predicted price category is: **{result}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
