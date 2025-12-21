import streamlit as st
import pandas as pd
import numpy as np

from prediction_helper import RULPredictor


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="RUL Prediction – Turbofan Engines",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("✈️ Remaining Useful Life (RUL) Prediction")
st.markdown(
    "Predict Remaining Useful Life of turbofan engines using a **leakage-safe classical ML model** "
    "trained on **NASA C-MAPSS FD001**."
)

# --------------------------------------------------
# Load Predictor (cached)
# --------------------------------------------------
@st.cache_resource
def load_predictor():
    return RULPredictor(
        "final_rul_model.joblib",
        "scaler.joblib",
        "features.joblib"
    )

predictor = load_predictor()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Input Options")

input_mode = st.sidebar.radio(
    "Choose input method:",
    ["Upload CSV", "Manual Input (Single Engine Snapshot)"]
)

# --------------------------------------------------
# CSV Upload Mode (UNCHANGED)
# --------------------------------------------------
if input_mode == "Upload CSV":
    st.subheader("📂 Upload CSV with Engine Features")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### Data Preview")
        st.dataframe(df.head())

        try:
            preds = predictor.predict(df)
            out = df.copy()
            out["Predicted_RUL"] = preds

            st.markdown("### Predictions")
            st.dataframe(out.head())

            st.download_button(
                "Download Predictions",
                out.to_csv(index=False),
                "rul_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(str(e))

# --------------------------------------------------
# Manual Input Mode (IMPROVED UI)
# --------------------------------------------------
else:
    st.subheader("🧮 Manual Input – Single Engine Snapshot")
    st.markdown(
        "Use **sliders OR typing** to enter feature values. "
        "Both inputs stay synchronized."
    )

    input_data = {}

    # Layout: 3 columns grid
    cols = st.columns(3)

    for i, feature in enumerate(predictor.features):
        with cols[i % 3]:
            st.markdown(f"**{feature}**")

            # Shared state key
            key = f"val_{feature}"

            if key not in st.session_state:
                st.session_state[key] = 0.0

            # Slider
            slider_val = st.slider(
                label="",
                min_value=-50.0,
                max_value=300.0,
                value=float(st.session_state[key]),
                step=0.1,
                key=f"{key}_slider"
            )

            # Number input
            number_val = st.number_input(
                label="",
                value=float(slider_val),
                step=0.1,
                key=f"{key}_number"
            )

            # Sync values
            st.session_state[key] = number_val
            input_data[feature] = number_val

    # --------------------------------------------------
    # Predict Button
    # --------------------------------------------------
    if st.button("🚀 Predict RUL"):
        input_df = pd.DataFrame([input_data])

        try:
            pred = predictor.predict(input_df)[0]

            st.success(f"🔧 **Predicted RUL: {pred:.2f} cycles**")

            st.markdown(
                """
                **Interpretation**
                - Higher RUL → healthier engine  
                - Lower RUL → closer to failure  
                """
            )

        except Exception as e:
            st.error(str(e))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "⚠️ **Note:** Model trained on NASA C-MAPSS FD001. "
    "Predictions are valid only under similar operating conditions."
)

