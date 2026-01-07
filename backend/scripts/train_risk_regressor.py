"""
Train Regression Model for Risk Score (0-100).

Instead of predicting risk class (safe/caution/high_risk), this model
predicts the actual risk_score which is more semantically meaningful.

Benefits:
- No "safe" label - model outputs a continuous risk score
- Better calibration - 0-100 scale is interpretable
- Can be combined with rule floors easily
"""

import json
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

BASE_DIR = Path(__file__).parent.parent  # backend/
DATA_DIR = BASE_DIR / "data" / "diabetic" / "training"
MODEL_DIR = BASE_DIR / "models"

# Drug class patterns for feature extraction
DRUG_CLASS_PATTERNS = {
    "hypoglycemia_risk": [
        "insulin",
        "sulfonylurea",
        "glipizide",
        "glyburide",
        "glimepiride",
    ],
    "hyperglycemia_risk": [
        "prednisone",
        "prednisolone",
        "dexamethasone",
        "methylprednisolone",
        "hydrocortisone",
        "cortisone",
        "corticosteroid",
    ],
    "thiazide": ["hydrochlorothiazide", "chlorthalidone", "indapamide", "metolazone"],
    "statin": [
        "atorvastatin",
        "simvastatin",
        "rosuvastatin",
        "pravastatin",
        "lovastatin",
    ],
    "ace_inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril", "benazepril"],
    "arb": ["losartan", "valsartan", "irbesartan", "olmesartan", "candesartan"],
    "beta_blocker": [
        "metoprolol",
        "atenolol",
        "propranolol",
        "carvedilol",
        "bisoprolol",
    ],
    "nsaid": ["ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac"],
    "antipsychotic": [
        "olanzapine",
        "clozapine",
        "quetiapine",
        "risperidone",
        "aripiprazole",
    ],
    "biguanide": ["metformin"],
    "sglt2": ["empagliflozin", "dapagliflozin", "canagliflozin"],
    "glp1": ["semaglutide", "liraglutide", "dulaglutide", "exenatide"],
    "dpp4": ["sitagliptin", "linagliptin", "saxagliptin", "alogliptin"],
    "thiazolidinedione": ["pioglitazone", "rosiglitazone"],
    "loop_diuretic": ["furosemide", "bumetanide", "torsemide"],
}


def get_drug_class_features(drug_name: str) -> List[float]:
    """Extract drug class one-hot features."""
    drug_lower = drug_name.lower()
    features = []

    for class_name, patterns in DRUG_CLASS_PATTERNS.items():
        match = 1.0 if any(p in drug_lower for p in patterns) else 0.0
        features.append(match)

    return features


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix for regression on risk_score."""

    features = []
    for _, row in df.iterrows():
        p = [
            row.get("age", 0) or 0,
            1 if str(row.get("gender", "")).lower().startswith("f") else 0,
            1 if str(row.get("gender", "")).lower().startswith("m") else 0,
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

        # Add drug class features
        drug_class_feats = get_drug_class_features(row.get("drug_name", ""))
        p.extend(drug_class_feats)

        features.append(p)

    X = np.array(features, dtype=np.float32)

    # Target is risk_score (0-100), not label (0-4)
    y = df["risk_score"].astype(float).values

    return X, y


def train():
    """Train regression model for risk_score."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(
        f"Risk score range: {train_df['risk_score'].min()} - {train_df['risk_score'].max()}"
    )
    print(f"Risk score mean: {train_df['risk_score'].mean():.2f}")

    # Build features
    X_train, y_train = build_features(train_df)
    X_val, y_val = build_features(val_df)
    X_test, y_test = build_features(test_df)

    # Scale features
    scaler = StandardScaler()
    n_patient_feats = 13 + len(DRUG_CLASS_PATTERNS)
    scaler.fit(X_train[:, :n_patient_feats])
    X_train[:, :n_patient_feats] = scaler.transform(X_train[:, :n_patient_feats])
    X_val[:, :n_patient_feats] = scaler.transform(X_val[:, :n_patient_feats])
    X_test[:, :n_patient_feats] = scaler.transform(X_test[:, :n_patient_feats])

    # High risk samples (risk_score > 50) get more weight
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train > 50] = 3.0  # 3x weight for high risk
    sample_weights[y_train > 80] = 5.0  # 5x weight for very high risk

    print(f"High risk samples (>50): {(y_train > 50).sum()}")
    print(f"Very high risk samples (>80): {(y_train > 80).sum()}")

    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_lambda=1.5,
        reg_alpha=0.1,
        min_child_weight=3,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predictions
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 100)  # Clamp to valid range

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # High-risk detection accuracy (risk_score > 50)
    actual_high_risk = y_test > 50
    predicted_high_risk = preds > 50
    high_risk_recall = (
        (actual_high_risk & predicted_high_risk).sum() / actual_high_risk.sum()
        if actual_high_risk.sum() > 0
        else 0
    )
    high_risk_precision = (
        (actual_high_risk & predicted_high_risk).sum() / predicted_high_risk.sum()
        if predicted_high_risk.sum() > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("REGRESSION MODEL EVALUATION")
    print("=" * 60)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print()
    print("High-Risk Detection (score > 50):")
    print(f"  Recall: {high_risk_recall:.3f}")
    print(f"  Precision: {high_risk_precision:.3f}")
    print("=" * 60)

    # Save artifacts
    artifact = {
        "model": model,
        "scaler": scaler,
        "model_type": "regression",
        "target": "risk_score",
        "model_version": f"diabetic_regression_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}",
    }

    joblib.dump(artifact, MODEL_DIR / "diabetic_risk_regressor.pkl")

    metrics = {
        "model_type": "regression",
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "high_risk_recall": float(high_risk_recall),
        "high_risk_precision": float(high_risk_precision),
    }

    with open(MODEL_DIR / "regression_training_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved regression model to {MODEL_DIR / 'diabetic_risk_regressor.pkl'}")


if __name__ == "__main__":
    train()
