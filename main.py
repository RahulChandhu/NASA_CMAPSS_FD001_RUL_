import streamlit as st
import pandas as pd
from prediction_helper import RULPredictor


# --------------------------------------------------
# Page config
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
    "Predict Remaining Useful Life of turbofan engines using a "
    "**leakage-safe classical ML model** trained on **NASA C-MAPSS FD001**."
)

# --------------------------------------------------
# Load predictor (cached)
# --------------------------------------------------
@st.cache_resource
def load_predictor():
    return RULPredictor(
        model_path="final_rul_model.joblib",
        scaler_path="scaler.joblib",
        features_path="features.joblib"
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
# CSV upload mode (unchanged)
# --------------------------------------------------
if input_mode == "Upload CSV":
    st.subheader("📂 Upload CSV with Engine Features")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
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
# Manual input mode (FIXED)
# --------------------------------------------------
else:
    st.subheader("🧮 Manual Input – Single Engine Snapshot")
    st.markdown(
        "Use **slider OR typing** to enter values. "
        "Both inputs are synchronized correctly."
    )

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(predictor.features):
        with cols[i % 3]:
            st.markdown(f"**{feature}**")

            # Initialize session state ONCE
            if feature not in st.session_state:
                st.session_state[feature] = 0.0

            # Slider (single source of truth)
            st.slider(
                label="",
                min_value=-50.0,
                max_value=300.0,
                step=0.1,
                key=feature
            )

            # Number input bound to SAME key
            st.number_input(
                label="",
                step=0.1,
                key=feature
            )

            # Read final value
            input_data[feature] = st.session_state[feature]

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    if st.button("🚀 Predict RUL"):
        input_df = pd.DataFrame([input_data])

        # Optional debug (can remove later)
        st.markdown("### 🔍 Model Input Snapshot")
        st.dataframe(input_df)

        try:
            pred = predictor.predict(input_df)[0]
            st.success(f"🔧 **Predicted RUL: {pred:.2f} cycles**")
        except Exception as e:
            st.error(str(e))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "⚠️ Model trained on NASA C-MAPSS FD001. "
    "Predictions valid only under similar operating conditions."
)
