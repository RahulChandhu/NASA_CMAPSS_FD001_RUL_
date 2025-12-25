# prediction_helper.py

import numpy as np
import joblib
import json
import os


MODEL_PATH = "final_hgb_model.joblib"
SCALER_PATH = "scaler.joblib"
FEATURE_PATH = "feature_names.json"


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Missing model file")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Missing scaler file")

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError("Missing feature_names.json")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURE_PATH, "r") as f:
        feature_names = json.load(f)

    # Hard validation
    if scaler.n_features_in_ != len(feature_names):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features "
            f"but feature list has {len(feature_names)}"
        )

    return model, scaler, feature_names


def predict_rul(input_features: np.ndarray) -> float:
    model, scaler, feature_names = load_artifacts()

    if input_features.shape != (1, len(feature_names)):
        raise ValueError(
            f"Expected input shape (1, {len(feature_names)}), "
            f"got {input_features.shape}"
        )

    scaled = scaler.transform(input_features)
    prediction = model.predict(scaled)

    # RUL must be non-negative
    return float(max(prediction[0], 0.0))

