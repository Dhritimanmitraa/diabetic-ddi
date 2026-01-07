"""
Diabetic ML Predictor V2 - Clinically-Trained Model

Uses the V2 model trained on rule engine labels with patient context.
This model predicts actual clinical risk, not just DDI frequency.
"""

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, List
import logging

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)

# Path to V2 model artifacts
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)
V2_MODEL_PATH = os.path.join(MODELS_DIR, "diabetic_risk_model_v2.joblib")
V2_TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_v2.joblib")
V2_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_v2.joblib")

RISK_SEVERITY = {
    "safe": "minor",
    "caution": "moderate",
    "high_risk": "major",
    "contraindicated": "contraindicated",
    "fatal": "fatal",
}

RISK_SCORE_MAP = {
    "safe": 0,
    "caution": 35,
    "high_risk": 60,
    "contraindicated": 85,
    "fatal": 100,
}


@dataclass
class DiabeticMLResultV2:
    """Result from V2 ML predictor."""

    risk_level: str
    probability: float
    probabilities: Dict[str, float]
    severity: str
    risk_score: float
    model_version: str
    is_rule_trained: bool = True  # Flag to indicate this model was trained on rules


class DiabeticMLPredictorV2:
    """
    V2 ML Predictor trained on clinically-meaningful labels.

    Features:
    - Trained on RULE ENGINE labels (weak supervision)
    - Uses PATIENT CONTEXT (eGFR, HbA1c, complications)
    - Uses DRUG CLASS features
    - Proper class balancing
    """

    def __init__(self):
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.classes = []
        self.is_loaded = False
        self.model_version = "2.0"
        self._load_lock = threading.Lock()
        self._load()

    def _load(self):
        """Load model artifacts."""
        try:
            if not os.path.exists(V2_MODEL_PATH):
                logger.warning(f"V2 model not found at {V2_MODEL_PATH}")
                return

            with self._load_lock:
                self.model = joblib.load(V2_MODEL_PATH)
                self.tfidf = joblib.load(V2_TFIDF_PATH)
                self.label_encoder = joblib.load(V2_LABEL_ENCODER_PATH)
                self.classes = list(self.label_encoder.classes_)
                self.is_loaded = True
                logger.info(f"Loaded V2 model: classes={self.classes}")
        except Exception as e:
            logger.error(f"Failed to load V2 model: {e}")
            self.is_loaded = False

    def _get_drug_class_features(self, drug_name: str) -> List[float]:
        """Extract drug class features (same logic as training)."""
        drug_lower = drug_name.lower()

        features = [0] * 15  # 15 drug class features

        # Hypoglycemia risk drugs
        hypoglycemia = [
            "glipizide",
            "glyburide",
            "glimepiride",
            "insulin",
            "sulfonylurea",
            "quinine",
        ]
        if any(h in drug_lower for h in hypoglycemia):
            features[0] = 1

        # Hyperglycemia risk (corticosteroids, antipsychotics, etc.)
        hyperglycemia = [
            "prednisone",
            "dexamethasone",
            "hydrocortisone",
            "methylprednisolone",
            "olanzapine",
            "quetiapine",
            "clozapine",
            "risperidone",
            "hydrochlorothiazide",
            "chlorthalidone",
        ]
        if any(h in drug_lower for h in hyperglycemia):
            features[1] = 1

        # Nephrotoxic
        nephrotoxic = [
            "ibuprofen",
            "naproxen",
            "diclofenac",
            "celecoxib",
            "meloxicam",
            "gentamicin",
            "tobramycin",
            "amikacin",
            "vancomycin",
            "amphotericin",
            "acyclovir",
            "tenofovir",
            "cisplatin",
            "methotrexate",
        ]
        if any(n in drug_lower for n in nephrotoxic):
            features[2] = 1

        # Hyperkalemia risk
        hyperkalemia = [
            "spironolactone",
            "eplerenone",
            "amiloride",
            "trimethoprim",
            "pril",
            "sartan",
        ]  # ACE/ARB
        if any(h in drug_lower for h in hyperkalemia):
            features[3] = 1

        # Hepatotoxic
        hepatotoxic = [
            "acetaminophen",
            "amiodarone",
            "valproic",
            "isoniazid",
            "methotrexate",
        ]
        if any(h in drug_lower for h in hepatotoxic):
            features[4] = 1

        # Cardioprotective
        cardioprotective = [
            "empagliflozin",
            "canagliflozin",
            "dapagliflozin",
            "liraglutide",
            "semaglutide",
            "dulaglutide",
        ]
        if any(c in drug_lower for c in cardioprotective):
            features[5] = 1

        # Masks hypoglycemia (beta blockers)
        beta_blockers = [
            "metoprolol",
            "atenolol",
            "propranolol",
            "carvedilol",
            "bisoprolol",
        ]
        if any(b in drug_lower for b in beta_blockers):
            features[6] = 1

        # Is corticosteroid
        if any(
            c in drug_lower
            for c in ["prednisone", "dexamethasone", "hydrocortisone", "methylpred"]
        ):
            features[7] = 1

        # Is NSAID
        if any(
            n in drug_lower
            for n in ["ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam"]
        ):
            features[8] = 1

        # Is ACE/ARB
        if "pril" in drug_lower or "sartan" in drug_lower:
            features[9] = 1

        # Is diuretic
        if any(
            d in drug_lower
            for d in [
                "furosemide",
                "hydrochlorothiazide",
                "chlorthalidone",
                "bumetanide",
            ]
        ):
            features[10] = 1

        # Is beta blocker
        if any(b in drug_lower for b in beta_blockers):
            features[11] = 1

        # Is sulfonylurea
        if any(
            s in drug_lower
            for s in ["glipizide", "glyburide", "glimepiride", "gliclazide"]
        ):
            features[12] = 1

        # Is SGLT2
        if "gliflozin" in drug_lower:
            features[13] = 1

        # Is GLP-1
        if "glutide" in drug_lower:
            features[14] = 1

        return features

    def predict(self, drug_name: str, patient: Dict) -> Optional[DiabeticMLResultV2]:
        """
        Predict drug risk for a diabetic patient.

        Args:
            drug_name: Name of drug to assess
            patient: Patient dict with labs (egfr, hba1c, potassium, alt) and complications

        Returns:
            DiabeticMLResultV2 with risk level and confidence
        """
        if not self.is_loaded:
            return None

        try:
            # Build feature vector
            # Patient numerical features
            patient_num = [
                patient.get("egfr") or 60,  # Default to moderate
                patient.get("potassium") or 4.5,
                patient.get("hba1c") or 7.0,
                patient.get("alt") or 25,
                patient.get("age") or 55,
            ]

            # Patient categorical features
            patient_cat = [
                1 if patient.get("has_nephropathy") else 0,
                1 if patient.get("has_cardiovascular") else 0,
                1 if patient.get("has_neuropathy") else 0,
                1 if patient.get("has_hypertension") else 0,
                1 if patient.get("diabetes_type") == "type_1" else 0,
            ]

            # Drug class features
            drug_class = self._get_drug_class_features(drug_name)

            # Drug name TF-IDF
            X_tfidf = self.tfidf.transform([drug_name.lower()])

            # Combine all features
            X = hstack(
                [
                    csr_matrix(np.array(patient_num).reshape(1, -1)),
                    csr_matrix(np.array(patient_cat).reshape(1, -1)),
                    csr_matrix(np.array(drug_class).reshape(1, -1)),
                    X_tfidf,
                ]
            )

            # Check if model needs dense array
            model_type = type(self.model).__name__
            if "Gradient" in model_type:
                X = X.toarray()

            # Predict
            proba = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            top_prob = float(proba[pred_idx])

            # Get class name
            risk_level = self.classes[pred_idx]

            # Build probability map
            prob_map = {
                self.classes[i]: float(proba[i]) for i in range(len(self.classes))
            }

            # Get severity and score
            severity = RISK_SEVERITY.get(risk_level, "moderate")
            risk_score = RISK_SCORE_MAP.get(risk_level, 35)

            # Adjust risk score based on confidence
            if top_prob < 0.7:
                # Low confidence: move toward middle
                if risk_level == "safe":
                    risk_score += 15
                elif risk_level in ["contraindicated", "fatal"]:
                    risk_score -= 10

            risk_score = max(0, min(100, risk_score))

            return DiabeticMLResultV2(
                risk_level=risk_level,
                probability=top_prob,
                probabilities=prob_map,
                severity=severity,
                risk_score=risk_score,
                model_version=self.model_version,
                is_rule_trained=True,
            )

        except Exception as e:
            logger.error(f"V2 prediction failed: {e}")
            return None


# Singleton instance
_predictor_v2: Optional[DiabeticMLPredictorV2] = None
_v2_lock = threading.Lock()


def get_diabetic_predictor_v2() -> Optional[DiabeticMLPredictorV2]:
    """Get singleton V2 predictor instance."""
    global _predictor_v2
    if _predictor_v2 is not None:
        return _predictor_v2
    with _v2_lock:
        if _predictor_v2 is None:
            _predictor_v2 = DiabeticMLPredictorV2()
    return _predictor_v2
