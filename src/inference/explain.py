import joblib
import pandas as pd
from pathlib import Path
import shap

from src.features.build_features import build_features, FEATURE_COLS, CAT_COLS

ARTIFACTS_DIR = Path("artifacts")


class Explainer:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.explainer = None

    def load(self):
        self.model = joblib.load(ARTIFACTS_DIR / "root_model.pkl")
        self.encoder = joblib.load(ARTIFACTS_DIR / "cat_encoder.pkl")
        self.explainer = shap.TreeExplainer(self.model)
        return self

    def explain_one(self, payload: dict) -> dict:
        df = pd.DataFrame([payload])
        df = build_features(df)

        for col in CAT_COLS:
            if col not in df.columns:
                df[col] = "UNKNOWN"

        X = df[FEATURE_COLS]
        X = self.encoder.transform(X)

        shap_values = self.explainer.shap_values(X)
        impacts = list(zip(FEATURE_COLS, shap_values[0]))
        impacts_sorted = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[:10]

        return {"top_feature_impacts": [{"feature": f, "shap_value": float(v)} for f, v in impacts_sorted]}