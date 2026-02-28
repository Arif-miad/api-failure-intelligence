import joblib
import pandas as pd
from pathlib import Path

from src.features.build_features import build_features, FEATURE_COLS, CAT_COLS
from src.features.encoders import CategoricalEncoder

ARTIFACTS_DIR = Path("artifacts")


class Predictor:
    def __init__(self):
        self.root_model = None
        self.retry_model = None
        self.resolution_model = None
        self.encoder: CategoricalEncoder | None = None

    def load(self):
        self.root_model = joblib.load(ARTIFACTS_DIR / "root_model.pkl")
        self.retry_model = joblib.load(ARTIFACTS_DIR / "retry_model.pkl")
        self.resolution_model = joblib.load(ARTIFACTS_DIR / "resolution_model.pkl")
        self.encoder = joblib.load(ARTIFACTS_DIR / "cat_encoder.pkl")
        return self

    def _prepare(self, payload: dict) -> pd.DataFrame:
        df = pd.DataFrame([payload])
        df = build_features(df)

        for col in CAT_COLS:
            if col not in df.columns:
                df[col] = "UNKNOWN"

        df = df[FEATURE_COLS]
        df = self.encoder.transform(df)
        return df

    def predict(self, payload: dict) -> dict:
        X = self._prepare(payload)

        root_pred = self.root_model.predict(X)[0]
        root_proba = float(max(self.root_model.predict_proba(X)[0]))

        retry_proba = float(self.retry_model.predict_proba(X)[0][1])

        res_pred = self.resolution_model.predict(X)[0]
        res_proba = float(max(self.resolution_model.predict_proba(X)[0]))

        return {
            "predicted_root_cause": str(root_pred),
            "root_cause_confidence": root_proba,
            "retry_success_probability": retry_proba,
            "recommended_action": str(res_pred),
            "resolution_confidence": res_proba,
        }