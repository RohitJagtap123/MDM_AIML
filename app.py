import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
# Ensure 'adaboost_model.pkl' and feature names were saved during training
model, feature_names = joblib.load("adaboost_model.pkl")  # Trained model and its feature names

# Streamlit app title
st.title("Bitcoin Price Prediction: High vs. Low")

# Introduction
st.write("""
This app predicts whether the Bitcoin price category is **High** or **Low** based on input features. 
Fill in the values below and click **Predict** to see the results.
""")

# Input form for custom data
st.header("Input Bitcoin Features")

# Create user input fields for features
year = st.number_input("Year", min_value=2000, max_value=2100, value=2023, step=1)
month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1, step=1)
open_price = st.number_input("Open Price (in USD)", value=0.0, step=0.01)
high_price = st.number_input("High Price (in USD)", value=0.0, step=0.01)
low_price = st.number_input("Low Price (in USD)", value=0.0, step=0.01)
volume = st.number_input("Volume (in millions)", value=0.0, step=1.0)

# Prepare input data
input_data = pd.DataFrame({
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price],
    'Volume': [volume]
})

# Ensure input_data matches model features
try:
    input_data = input_data[feature_names]
except KeyError:
    st.error("Feature mismatch! Ensure the input data matches the model's features.")

# Display input data (optional)
st.write("Input Data:")
st.write(input_data)

# Predict button
if st.button("Predict"):
    try:
        # Use the model to make a prediction
        prediction = model.predict(input_data)
        result = "High" if prediction[0] == 1 else "Low"
        
        # Display the result
        st.success(f"The predicted price category is: **{result}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
