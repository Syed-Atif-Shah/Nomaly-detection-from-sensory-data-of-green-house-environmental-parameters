import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained models with proper error handling
try:
    anomaly_model = joblib.load('anomaly_detection_model.pkl')  # Anomaly Detection Model
    crop_model = joblib.load('crop_suggestion_model.pkl')      # Crop Environment Suggestion Model
except (FileNotFoundError, joblib.externals.loky.process_executor.TerminatedWorkerError, EOFError) as e:
    st.error(f"Model file error: {e}. Please ensure models exist and are not corrupted.")
    st.stop()

# List of available crops
crops = ['Tomato', 'Rice', 'Wheat', 'Cucumber']

# Streamlit app layout
st.title('Anomaly Detection and Crop Optimization')
st.write('Enter environmental data to detect anomalies and get crop optimization suggestions.')

# Crop selection input
crop = st.selectbox('Select Crop', crops)

# Input for environmental parameters
co = st.number_input('CO (Carbon Monoxide)', min_value=0.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, step=0.1)
light = st.number_input('Light (Lux)', min_value=0.0, step=0.1)
lpg = st.number_input('LPG', min_value=0.0, step=0.1)
motion = st.selectbox('Motion', [0, 1])  # Binary input (0 or 1)
smoke = st.selectbox('Smoke', [0, 1])    # Binary input (0 or 1)
temp = st.number_input('Temperature (°C)', min_value=-50.0, max_value=50.0, step=0.1)

# Submit button to trigger predictions
if st.button('Submit'):
    # Collect parameters for anomaly detection
    anomaly_input_data = np.array([co, humidity, light, lpg, motion, smoke, temp], dtype=np.float32).reshape(1, -1)
    
    try:
        # Step 1: Anomaly Detection
        anomaly_prediction = anomaly_model.predict(anomaly_input_data)
        if isinstance(anomaly_prediction, np.ndarray) and anomaly_prediction.shape[0] > 0:
            anomaly_prediction = anomaly_prediction[0]  # Extract scalar value
        else:
            raise ValueError("Invalid anomaly model output.")

        if anomaly_prediction == -1:
            st.error("Input parameters detected as anomalies. Taking corrective actions...")
            
            # Prepare input for crop environment suggestion model
            crop_mapping = {'Tomato': 109, 'Rice': 35, 'Wheat': 12, 'Cucumber': 57}
            crop_id = crop_mapping.get(crop, -1)
            crop_input_data = pd.DataFrame(
                [[crop_id, co, humidity, temp, light, smoke]],
                columns=['Crop', 'optimal CO', 'Optimal Humidity', 'Optimal Temp', 'Optimal Light', 'Optimal Smoke']
            )
            
            # Debugging step: Print crop input data
            st.write("Crop Input Data:")
            st.write(crop_input_data)
            
            # Step 2: Crop Environment Suggestion Model
            crop_suggestion = crop_model.predict(crop_input_data)
            
            # Debugging step: Print crop suggestion output
            st.write("Crop Suggestion Output:")
            st.write(crop_suggestion)
            
            if isinstance(crop_suggestion, np.ndarray) and crop_suggestion.shape[0] > 0:
                optimal_params = crop_suggestion[0]
                
                # Debugging step: Print optimal parameters
                st.write("Optimal Parameters:")
                st.write(optimal_params)
                
                if isinstance(optimal_params, (list, np.ndarray)) and len(optimal_params) == 6:
                    optimal_co, optimal_humidity, optimal_temp, optimal_light, optimal_smoke, suggested_action = optimal_params
                    
                    # Generate action recommendations
                    actions_needed = []
                    if temp > optimal_temp:
                        actions_needed.append("Reduce temperature (e.g., turn on cooling fan).")
                    if humidity < optimal_humidity:
                        actions_needed.append("Increase humidity (e.g., use humidifier).")
                    if light < optimal_light:
                        actions_needed.append("Increase light intensity (e.g., turn on grow lights).")
                    if smoke > optimal_smoke:
                        actions_needed.append("Reduce smoke levels (e.g., improve ventilation).")
                    
                    if actions_needed:
                        st.warning("Suggested corrective actions:")
                        for action in actions_needed:
                            st.write(f"- {action}")
                    else:
                        st.success("No corrective actions needed. Parameters are optimal.")
                else:
                    st.error("Crop model output format is incorrect.")
            else:
                st.error("Unable to fetch optimized conditions for the selected crop.")
        else:
            st.success("Input parameters are normal.")
            st.write("No corrective actions needed. Parameters are optimal.")

    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")
