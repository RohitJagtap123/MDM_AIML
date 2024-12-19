import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("adaboost_model.pkl")  # Replace with your actual model file path

# Set up Streamlit app
st.title("Bitcoin Price Prediction: Adj Close")
st.write("""
This application predicts the **Adjusted Close (Adj Close)** price of Bitcoin based on the following input features:
- Open Price
- High Price
- Low Price
- Volume
""")

# Input form for user inputs
st.header("Input Features")

# Input fields for the features
open_price = st.number_input("Open Price (in USD)", value=0.0, step=0.01)
high_price = st.number_input("High Price (in USD)", value=0.0, step=0.01)
low_price = st.number_input("Low Price (in USD)", value=0.0, step=0.01)
close_price = st.number_input("Close Price (in USD)", value=0.0, step=0.01)
volume = st.number_input("Volume (in millions)", value=0.0, step=1.0)

# Organize inputs into a DataFrame
input_data = pd.DataFrame({
    'Open': [open_price],
    'High': [high_price],
    'Low': [low_price],
    'Close': [close_price],
    'Volume': [volume]
})

# Display the input data (optional)
st.write("Input Data:")
st.write(input_data)

# Prediction logic
if st.button("Predict"):
    try:
        # Ensure input data matches model's feature structure
        prediction = model.predict(input_data)
        predicted_price = round(prediction[0], 2)

        # Display prediction
        st.success(f"The predicted Adj Close price is: **${predicted_price}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
