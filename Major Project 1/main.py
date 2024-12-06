import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime

# Load the pre-trained model and scalers
model = load_model("lstm_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Interpretation function
def interpret_climatic_conditions(predictions, columns):
    conditions = []
    predicted_values = dict(zip(columns, predictions))

    temp = predicted_values["t2m"] - 273.15  # Convert to Celsius
    if temp > 35:
        conditions.append("🔥 Heatwave warning")
    elif 0 <= temp <= 35:
        conditions.append("🌡️ Normal temperature")
    elif temp < 0:
        conditions.append("❄️ Frost warning")

    precip = predicted_values["tp"]
    if precip > 50:
        conditions.append("🌧️ Heavy rainfall alert")
    elif 1 < precip <= 50:
        conditions.append("🌤️ Clear weather with no chance of rainfall.")
    elif precip <= 1:
        conditions.append("☀️ Dry conditions")

    wind_speed = (predicted_values["u10"] ** 2 + predicted_values["v10"] ** 2) ** 0.5
    if wind_speed > 50:
        conditions.append("💨 Strong wind advisory")
    else:
        conditions.append("🍃 Normal wind speed")

    pressure = predicted_values["sp"]
    if pressure < 1000:
        conditions.append("🌪️ Low pressure: Possible storm")
    else:
        conditions.append("🌈 Normal pressure: No sign of storm")

    return conditions, temp

# Prediction function
def predict_for_date(date, hour):
    day_of_year = date.timetuple().tm_yday
    input_features = np.array([[day_of_year, hour]])
    input_features_scaled = scaler_X.transform(input_features)
    input_features_scaled = np.expand_dims(input_features_scaled, axis=1)
    predictions_scaled = model.predict(input_features_scaled)
    st.write("Raw Predictions (Scaled):", predictions_scaled)  # Debugging
    predictions_rescaled = scaler_y.inverse_transform(predictions_scaled)
    st.write("Rescaled Predictions:", predictions_rescaled)  # Debugging
    return predictions_rescaled[0]

st.set_page_config("AdaptiPlan")

# Modern Streamlit UI
st.markdown(
    """
    <style>
    
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    body {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        color: #333;
        font-size: 3rem;
        margin-bottom: 14px;
    }
    .description {
        text-align: center;
        font-family: 'Poppins', sans-serif;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    .insights {
        background-color: #e9f5ff;
        border-left: 5px solid #0078d4;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        font-family: 'Poppins', sans-serif;
        color: #333;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'><strong>AdaptiPlan 🌤️</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='description'>Your AI-powered weather and climatic insights tool 🌍</div>",
    unsafe_allow_html=True,
)

# Sidebar for input
st.sidebar.header("Input Parameters")
user_date = st.sidebar.date_input("📅 Select Date", datetime.date.today())
user_hour = st.sidebar.slider("🕒 Select Hour", 0, 23, 12)

target_columns = ["d2m", "t2m", "tp", "sp", "u10", "v10"]

if st.sidebar.button("🌟 Predict"):
    st.write(f"Processing input for **{user_date}, {user_hour}:00**...")

    user_prediction = predict_for_date(user_date, user_hour)
    conditions, temperature_celsius = interpret_climatic_conditions(
        user_prediction, target_columns
    )

    st.subheader(f"📋 Predicted Weather for {user_date}, {user_hour}:00")
    # st.write(f"🌡️ **Temperature (Celsius): {temperature_celsius:.2f}**")

    # st.markdown("<div class='insights'>", unsafe_allow_html=True)
    st.write("### Climatic Insights:")
    for condition in conditions:
        st.write(f"- {condition}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.balloons()
