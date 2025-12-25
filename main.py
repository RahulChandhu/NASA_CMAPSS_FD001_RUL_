# main.py

import streamlit as st
import numpy as np
from prediction_helper import load_artifacts, predict_rul

st.set_page_config(
    page_title="RUL Prediction | NASA C-MAPSS FD001",
    layout="wide"
)

# ---------------- LOAD ARTIFACTS ----------------
model, scaler, FEATURE_NAMES = load_artifacts()

# ---------------- HEADER ----------------
st.title("🛠️ Turbofan Engine RUL Prediction")
st.caption("NASA C-MAPSS FD001 · Leakage-safe ML model")

st.markdown(
    """
This app predicts **Remaining Useful Life (RUL)** of a turbofan engine  
using the **exact same features and scaling** as the trained model.

⚠️ Predictions near failure have higher uncertainty.
"""
)

st.divider()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Engine Sensor Inputs")
st.sidebar.caption("Values must match training scale")

inputs = []

for feature in FEATURE_NAMES:
    value = st.sidebar.number_input(
        label=feature,
        value=0.0,
        format="%.6f",
        key=f"input_{feature}"
    )
    inputs.append(value)

input_array = np.array(inputs, dtype=float).reshape(1, -1)

# ---------------- PREDICTION ----------------
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    predict_btn = st.button("🔍 Predict RUL", use_container_width=True)

with col2:
    if predict_btn:
        try:
            rul = predict_rul(input_array)

            st.success("Prediction completed")

            st.metric(
                label="Estimated Remaining Useful Life (cycles)",
                value=f"{rul:.2f}"
            )

            st.warning(
                "This prediction is a decision-support signal, "
                "not a guarantee of failure time."
            )

        except Exception as e:
            st.error("Prediction failed due to invalid input or setup.")
            st.exception(e)

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    f"Model input features: {len(FEATURE_NAMES)} · "
    "HistGradientBoostingRegressor · "
    "FD001 dataset"
)

