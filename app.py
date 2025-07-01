# Enhanced Streamlit App for Week 3
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# 🎯 Title
st.set_page_config(page_title="Water Pollutants Predictor", layout="centered")
st.title("💧 Water Pollutants Predictor")
st.write("Predict the levels of water pollutants based on the Year and Station ID.")

# ⏩ Load Model and Columns
@st.cache_resource
def load_model():
    model = joblib.load("pollution_model.pkl")
    model_cols = joblib.load("model_columns.pkl")
    return model, model_cols
model, model_cols = load_model()
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# ✅ Extract valid station IDs from model columns
valid_ids = sorted([col.split('_')[1] for col in model_cols if col.startswith('id_')])
valid_ids.insert(0, '1')  # Include base column 'id' as '1'

# 🧑‍💻 User Inputs
col1, col2 = st.columns(2)
with col1:
    year_input = st.number_input("📅 Enter Year", min_value=2000, max_value=2100, value=2022)
with col2:
    station_id = st.selectbox("🏭 Select Station ID", options=valid_ids)

# 🚀 Predict Button
if st.button("🔍 Predict"):
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    # 🧩 Align columns
    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_cols]

    # 🔮 Predict
    predicted_pollutants = model.predict(input_encoded)[0]

    # 📈 Output: Metrics
    st.subheader(f"Predicted Pollutant Levels for Station {station_id} in {year_input}")
    cols = st.columns(3)
    for i, (p, val) in enumerate(zip(pollutants, predicted_pollutants)):
        cols[i % 3].metric(label=p, value=f"{val:.2f}")

    # 📊 Output: Bar Chart
    st.markdown("### 📉 Visualized Prediction")
    fig, ax = plt.subplots()
    ax.bar(pollutants, predicted_pollutants, color='steelblue')
    ax.set_ylabel("Pollutant Concentration")
    ax.set_title("Predicted Water Pollutants")
    st.pyplot(fig)