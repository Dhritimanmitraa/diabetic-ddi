"""
Diabetic-specific ML predictor for drug risk.

Loads the trained model artifact produced by train_diabetic_ml.py and
provides a predict() helper compatible with service usage.
"""

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

import joblib
import numpy as np

DEFAULT_MODEL_PATH = os.environ.get("DIABETIC_MODEL_PATH", "./models/diabetic_risk_model.pkl")

RISK_SEVERITY = {
    "safe": "minor",
    "caution": "moderate",
    "high_risk": "major",
    "contraindicated": "contraindicated",
    "fatal": "fatal",
}


def hash_text(text: str, n_features: int) -> np.ndarray:
    vec = np.zeros(n_features, dtype=np.float32)
    if not isinstance(text, str):
        return vec
    lower = text.lower()
    for idx, ch in enumerate(lower):
        bucket = (hash(ch + str(idx)) % n_features)
        vec[bucket] += 1.0
    return vec


def build_vector(patient: Dict, drug_name: str, scaler, hash_size: int) -> np.ndarray:
    num_features = [
        patient.get("age") or 0,
        1 if str(patient.get("gender", "")).lower().startswith("f") else 0,
        1 if str(patient.get("gender", "")).lower().startswith("m") else 0,
        patient.get("creatinine") or 0,
        patient.get("potassium") or 0,
        patient.get("fasting_glucose") or 0,
        1 if patient.get("has_nephropathy") else 0,
        1 if patient.get("has_retinopathy") else 0,
        1 if patient.get("has_neuropathy") else 0,
        1 if patient.get("has_cardiovascular") else 0,
        1 if patient.get("has_hypertension") else 0,
        1 if patient.get("has_hyperlipidemia") else 0,
        1 if patient.get("has_obesity") else 0,
    ]
    drug_vec = hash_text(drug_name or "", hash_size)
    full = np.concatenate([np.array(num_features, dtype=np.float32), drug_vec])
    # Scale numeric part
    full[:13] = scaler.transform(full[:13].reshape(1, -1))[0]
    return full.reshape(1, -1)


@dataclass
class DiabeticMLResult:
    risk_level: str
    probability: float
    probabilities: Dict[str, float]
    severity: str
    risk_score: float
    model_version: Optional[str]


class DiabeticMLPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.hash_size = 48
        self.risk_to_int = {}
        self.int_to_risk = {}
        self.model_version = None
        self.is_loaded = False
        self._load_lock = threading.Lock()
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        with self._load_lock:
            artifact = joblib.load(self.model_path)
            self.model = artifact["model"]
            self.scaler = artifact["scaler"]
            self.hash_size = artifact.get("hash_size", 48)
            self.risk_to_int = artifact.get("risk_to_int", {})
            self.int_to_risk = artifact.get("int_to_risk", {})
            self.model_version = artifact.get("model_version")
            self.is_loaded = True

    def predict(self, drug_name: str, patient: Dict) -> Optional[DiabeticMLResult]:
        if not self.is_loaded:
            return None
        vec = build_vector(patient, drug_name, self.scaler, self.hash_size)
        proba = self.model.predict_proba(vec)[0]
        pred_idx = int(np.argmax(proba))
        risk_level = self.int_to_risk.get(pred_idx, "caution")
        prob_map = {self.int_to_risk.get(i, str(i)): float(p) for i, p in enumerate(proba)}
        top_prob = float(proba[pred_idx])
        severity = RISK_SEVERITY.get(risk_level, "moderate")
        risk_score = round(top_prob * 100, 2)
        return DiabeticMLResult(
            risk_level=risk_level,
            probability=top_prob,
            probabilities=prob_map,
            severity=severity,
            risk_score=risk_score,
            model_version=self.model_version,
        )


_predictor: Optional[DiabeticMLPredictor] = None
_pred_lock = threading.Lock()


def get_diabetic_predictor(model_path: str = DEFAULT_MODEL_PATH) -> DiabeticMLPredictor:
    global _predictor
    if _predictor and _predictor.is_loaded:
        return _predictor
    with _pred_lock:
        if _predictor is None:
            _predictor = DiabeticMLPredictor(model_path)
    return _predictor


