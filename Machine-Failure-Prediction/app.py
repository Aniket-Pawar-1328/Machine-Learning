import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open('best_estimator.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_machine_status(model, scaler, num_col, Qual, ambient_t, process_t, rot_speed, torq, Tool_w):
    input_data = {
        "Quality": [Qual],
        "Ambient_temp": [ambient_t],
        "Process_temp": [process_t],
        "Rotation_speed": [rot_speed],
        "Torque": [torq],
        "Tool_wear": [Tool_w]
    }

    input_df = pd.DataFrame(input_data)
    input_df_encoded = pd.get_dummies(input_df, columns=['Quality'])
    input_df_encoded[num_col] = scaler.transform(input_df_encoded[num_col])

    for col in ['Quality_L', 'Quality_M', 'Quality_H']:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    input_for_prediction = input_df_encoded.values.reshape(1, -1)
    prediction = model.predict(input_for_prediction)
    
    if prediction[0] == 1:
        return "Machine is faulty"
    else:
        return "Machine is not faulty"

st.title("Machine Fault Prediction")

Qual = st.selectbox("Select Quality (M, L, H):", ['M', 'L', 'H'])

ambient_t = st.number_input("Enter Ambient Temperature:", min_value=0.0, max_value=100.0, step=0.1)
process_t = st.number_input("Enter Process Temperature:", min_value=0.0, max_value=100.0, step=0.1)
rot_speed = st.number_input("Enter Rotation Speed:", min_value=0.0, step=0.1)
torq = st.number_input("Enter Torque:", min_value=0.0, step=0.1)
Tool_w = st.number_input("Enter Tool Wear:", min_value=0.0, step=0.1)

num_col = ['Ambient_temp', 'Process_temp', 'Rotation_speed', 'Torque', 'Tool_wear']

if st.button("Predict Machine Status"):
    result = predict_machine_status(model, scaler, num_col, Qual, ambient_t, process_t, rot_speed, torq, Tool_w)
    st.success(result)
