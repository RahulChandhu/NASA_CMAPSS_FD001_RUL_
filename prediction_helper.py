import pandas as pd
import joblib


class RULPredictor:
    """
    Loads trained artifacts and performs RUL prediction.
    """

    def __init__(self, model_path, scaler_path, features_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return df[self.features]

    def predict(self, df: pd.DataFrame):
        X = self._validate_input(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
