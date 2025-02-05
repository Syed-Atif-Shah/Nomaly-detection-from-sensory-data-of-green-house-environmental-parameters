import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
try:
    model = joblib.load('anomaly_detection_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'anomaly_detection_model.pkl' exists in the correct directory.")
    st.stop()

# Streamlit app layout
st.title('Anomaly Detection with Isolation Forest')
st.write('Upload your temperature data to detect anomalies.')

# Upload the CSV file with sensor data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Uploaded Data:")
    st.write(df.head())

    # Show columns to verify if the correct columns are being uploaded
    st.write("Columns in uploaded CSV:")
    st.write(df.columns)
    
    # Validate the input data
    expected_columns = ['feature1', 'feature2', 'feature3']  # Replace with actual feature names
    if not all(col in df.columns for col in expected_columns):
        st.error(f"CSV file must contain columns: {expected_columns}")
    else:
        # Make predictions using the model
        try:
            predictions = model.predict(df[expected_columns])
            st.write(f"Model predictions: {predictions}")  # Debugging statement
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.stop()

        # Add predictions to the DataFrame
        df['anomaly'] = predictions
        
        # Display the prediction results
        st.write("Prediction Results:")
        st.write(df[['anomaly']])
        
        # Show anomalies
        anomalies = df[df['anomaly'] == -1]
        st.write("Anomalies detected:")
        st.write(anomalies)
else:
    st.write("Please upload a CSV file.")

