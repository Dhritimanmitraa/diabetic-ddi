"""
Train a diabetic-specific drug risk model using pseudo-labels from rules.

Steps:
- Load train/val/test CSVs produced by build_diabetic_pseudolabels.py
- Vectorize patient context + drug name
- Train multi-class classifier (XGBoost)
- Calibrate probabilities
- Persist model artifact and training summary
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "diabetic" / "training"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "diabetic_risk_model.pkl"
RESULTS_PATH = MODEL_DIR / "diabetic_training_results.json"

RISK_LEVELS = ["safe", "caution", "high_risk", "contraindicated", "fatal"]
RISK_TO_INT = {k: i for i, k in enumerate(RISK_LEVELS)}
INT_TO_RISK = {i: k for k, i in RISK_TO_INT.items()}


def hash_text(text: str, n_features: int = 48) -> np.ndarray:
    """Simple character hash embedding for small vocabulary drug names."""
    vec = np.zeros(n_features, dtype=np.float32)
    if not isinstance(text, str):
        return vec
    lower = text.lower()
    for idx, ch in enumerate(lower):
        bucket = (hash(ch + str(idx)) % n_features)
        vec[bucket] += 1.0
    return vec


def build_feature_vector(row: pd.Series, hash_size: int = 48) -> np.ndarray:
    num_features = [
        row.get("age", 0) or 0,
        1 if (str(row.get("gender", "")).lower().startswith("f")) else 0,
        1 if (str(row.get("gender", "")).lower().startswith("m")) else 0,
        float(row.get("creatinine", 0) or 0),
        float(row.get("potassium", 0) or 0),
        float(row.get("fasting_glucose", 0) or 0),
        int(bool(row.get("has_nephropathy", False))),
        int(bool(row.get("has_retinopathy", False))),
        int(bool(row.get("has_neuropathy", False))),
        int(bool(row.get("has_cardiovascular", False))),
        int(bool(row.get("has_hypertension", False))),
        int(bool(row.get("has_hyperlipidemia", False))),
        int(bool(row.get("has_obesity", False))),
    ]
    drug_name_val = row.get("drug_name") or ""
    drug_vec = hash_text(str(drug_name_val), n_features=hash_size)
    return np.concatenate([np.array(num_features, dtype=np.float32), drug_vec])


def build_matrices(df: pd.DataFrame, hash_size: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([build_feature_vector(row, hash_size) for _, row in df.iterrows()])
    y = np.asarray(df["label"].astype(int).values)
    return X, y


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, val, test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    params = dict(
        objective="multi:softprob",
        num_class=len(RISK_LEVELS),
        n_estimators=160,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="mlogloss",
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def calibrate_model(model: xgb.XGBClassifier, X_val: np.ndarray, y_val: np.ndarray):
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(X_val, y_val)
    return calibrator


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = load_data()

    hash_size = 48
    X_train, y_train = build_matrices(train_df, hash_size)
    X_val, y_val = build_matrices(val_df, hash_size)
    X_test, y_test = build_matrices(test_df, hash_size)

    # Standardize numeric features lightly (first 13 columns) and leave hashes raw
    scaler = StandardScaler()
    scaler.fit(X_train[:, :13])
    for arr in [X_train, X_val, X_test]:
        arr[:, :13] = scaler.transform(arr[:, :13])

    model = train_model(X_train, y_train)
    calibrated = calibrate_model(model, X_val, y_val)

    proba = calibrated.predict_proba(X_test)
    preds = np.argmax(proba, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "classes": INT_TO_RISK,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)  # type: ignore
    metrics["classification_report"] = report

    artifact = {
        "model": calibrated,
        "scaler": scaler,
        "hash_size": hash_size,
        "risk_to_int": RISK_TO_INT,
        "int_to_risk": INT_TO_RISK,
        "model_version": f"diabetic_mimic_demo_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}",
    }

    joblib.dump(artifact, MODEL_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", MODEL_PATH)
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])


if __name__ == "__main__":
    main()


