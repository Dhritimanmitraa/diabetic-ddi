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
from sklearn.utils.class_weight import compute_class_weight
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
    drug_vec = hash_text(row.get("drug_name", ""), n_features=hash_size)
    return np.concatenate([np.array(num_features, dtype=np.float32), drug_vec])


def build_matrices(df: pd.DataFrame, hash_size: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([build_feature_vector(row, hash_size) for _, row in df.iterrows()])
    y = df["label"].astype(int).values
    return X, y


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, val, test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    """Train XGBoost with class weights to handle imbalance."""
    
    # Compute class weights to handle severe imbalance (92.7% safe)
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    
    # Create sample weights array
    sample_weights = np.array([class_weight_dict[int(y)] for y in y_train])
    
    print(f"Class weights: {class_weight_dict}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    params = dict(
        objective="multi:softprob",
        num_class=len(RISK_LEVELS),
        n_estimators=200,  # Increased for better learning
        max_depth=8,  # Deeper trees for complex patterns
        learning_rate=0.08,  # Slightly lower for better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_lambda=1.5,  # Stronger regularization
        reg_alpha=0.1,
        eval_metric="mlogloss",
        min_child_weight=3,  # Prevent overfitting to majority
        scale_pos_weight=1,  # We use sample_weight instead
    )
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def calibrate_model(model: xgb.XGBClassifier, X_val: np.ndarray, y_val: np.ndarray):
    """Calibrate the model, return the base model if calibration fails."""
    # Try calibration, but if it fails, just return the base model
    try:
        # Use cross-validation for calibration (more robust)
        calibrator = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrator.fit(X_val, y_val)
        return calibrator
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Warning: Calibration failed ({e}), using base model without calibration")
        return model  # Return base model if calibration fails


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

    # Test predictions
    try:
        proba = calibrated.predict_proba(X_test)
        preds = np.argmax(proba, axis=1)
    except Exception as e:
        print(f"Warning: predict_proba failed ({e}), using predict instead")
        preds = calibrated.predict(X_test)
        # Create dummy probabilities for metrics
        proba = np.zeros((len(preds), len(RISK_LEVELS)))
        for i, p in enumerate(preds):
            proba[i, p] = 1.0

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "weighted_f1": float(f1_score(y_test, preds, average="weighted")),
        "classes": INT_TO_RISK,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "training_method": "class_weighted",  # Document that we used class weights
    }
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    metrics["classification_report"] = report
    
    # Track critical metrics: recall on risky classes
    # These are more important than overall accuracy
    risky_classes = ["high_risk", "contraindicated", "fatal"]
    for risk_class in risky_classes:
        class_idx = str(RISK_TO_INT[risk_class])
        if class_idx in report:
            metrics[f"{risk_class}_recall"] = report[class_idx].get("recall", 0)
            metrics[f"{risk_class}_precision"] = report[class_idx].get("precision", 0)
            metrics[f"{risk_class}_f1"] = report[class_idx].get("f1-score", 0)

    # Save both the calibrated model and the base model for compatibility
    artifact = {
        "model": calibrated,  # Calibrated model (preferred)
        "base_model": model,  # Base XGBClassifier (fallback)
        "scaler": scaler,
        "hash_size": hash_size,
        "risk_to_int": RISK_TO_INT,
        "int_to_risk": INT_TO_RISK,
        "model_version": f"diabetic_class_weighted_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}",
    }

    joblib.dump(artifact, MODEL_PATH)
    with open(RESULTS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print()
    print("Critical Risk Class Performance:")
    for risk_class in risky_classes:
        recall = metrics.get(f"{risk_class}_recall", 0)
        precision = metrics.get(f"{risk_class}_precision", 0)
        print(f"  {risk_class}: Recall={recall:.3f}, Precision={precision:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


