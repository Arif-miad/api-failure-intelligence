import json
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.features.build_features import build_features, FEATURE_COLS, CAT_COLS
from src.features.encoders import CategoricalEncoder

ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path(r"F:\api-failure-intelligence\data\raw\dataset.csv")


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    y = df["root_cause"].astype(str)

    X = df[FEATURE_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str).fillna("UNKNOWN")

    encoder = CategoricalEncoder(CAT_COLS).fit(X)
    X_enc = encoder.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.1,
        loss_function="MultiClass",
        verbose=50,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    joblib.dump(model, ARTIFACTS_DIR / "root_model.pkl")
    joblib.dump(encoder, ARTIFACTS_DIR / "cat_encoder.pkl")

    with open(ARTIFACTS_DIR / "metrics_root.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Root cause model trained & saved.")


if __name__ == "__main__":
    main()