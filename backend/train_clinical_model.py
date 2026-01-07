"""
Train ML Model with Clinically-Meaningful Labels (V2)

This script fixes the fundamental problem with the previous model:
- Labels are generated from the CLINICAL RULE ENGINE (weak supervision)
- Features include PATIENT CONTEXT (eGFR, HbA1c, complications)
- Proper class balancing is applied
- Drug class features are extracted from rules.py knowledge

The model learns to predict: "Given this drug + patient, what is the clinical risk?"
Instead of: "Does this drug have frequent DDI labels?" (useless)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from app.diabetic.rules import DiabeticDrugRules

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================================
# STEP 1: Define Drug List + Synthetic Patient Profiles
# ============================================================================

# Key drugs to train on (from rules.py + common diabetic medications)
TRAINING_DRUGS = [
    # Diabetes medications
    "metformin",
    "glipizide",
    "glyburide",
    "glimepiride",
    "pioglitazone",
    "sitagliptin",
    "linagliptin",
    "empagliflozin",
    "canagliflozin",
    "dapagliflozin",
    "liraglutide",
    "semaglutide",
    "dulaglutide",
    "insulin",
    "insulin glargine",
    # Hyperglycemia risk drugs
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
    "furosemide",
    "tacrolimus",
    "cyclosporine",
    "sirolimus",
    # Nephrotoxic drugs
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
    "cidofovir",
    "cisplatin",
    "methotrexate",
    "lithium",
    # Cardiovascular / Antihypertensives
    "lisinopril",
    "enalapril",
    "ramipril",
    "losartan",
    "valsartan",
    "amlodipine",
    "nifedipine",
    "diltiazem",
    "verapamil",
    "metoprolol",
    "atenolol",
    "carvedilol",
    "propranolol",
    "spironolactone",
    "eplerenone",
    "amiloride",
    # Hypoglycemia risk drugs
    "warfarin",
    "fluconazole",
    "ciprofloxacin",
    "levofloxacin",
    "clarithromycin",
    "trimethoprim",
    "pentamidine",
    "quinine",
    # Hepatotoxic
    "acetaminophen",
    "amiodarone",
    "valproic acid",
    "isoniazid",
    # Anticoagulants
    "aspirin",
    "clopidogrel",
    "rivaroxaban",
    "apixaban",
    "dabigatran",
    # Others
    "gabapentin",
    "pregabalin",
    "duloxetine",
    "tramadol",
    "morphine",
    "omeprazole",
    "pantoprazole",
    "ranitidine",
    "famotidine",
    "atorvastatin",
    "rosuvastatin",
    "simvastatin",
    "pravastatin",
    "levothyroxine",
    "allopurinol",
    "febuxostat",
]

# Synthetic patient profiles covering the risk spectrum
PATIENT_PROFILES = [
    # Healthy diabetic - most drugs safe
    {
        "name": "healthy_t2_controlled",
        "diabetes_type": "type_2",
        "years_with_diabetes": 5,
        "age": 45,
        "hba1c": 6.8,
        "fasting_glucose": 120,
        "egfr": 90,
        "creatinine": 0.9,
        "potassium": 4.2,
        "alt": 25,
        "ast": 22,
        "has_nephropathy": False,
        "has_retinopathy": False,
        "has_neuropathy": False,
        "has_cardiovascular": False,
        "has_hypertension": False,
    },
    # Moderate kidney disease
    {
        "name": "moderate_ckd",
        "diabetes_type": "type_2",
        "years_with_diabetes": 10,
        "age": 62,
        "hba1c": 7.5,
        "fasting_glucose": 150,
        "egfr": 45,
        "creatinine": 1.6,
        "potassium": 4.8,
        "alt": 30,
        "ast": 28,
        "has_nephropathy": True,
        "has_retinopathy": False,
        "has_neuropathy": True,
        "has_cardiovascular": True,
        "has_hypertension": True,
    },
    # Severe kidney disease - many drugs contraindicated
    {
        "name": "severe_ckd",
        "diabetes_type": "type_2",
        "years_with_diabetes": 15,
        "age": 70,
        "hba1c": 8.2,
        "fasting_glucose": 180,
        "egfr": 22,
        "creatinine": 3.0,
        "potassium": 5.4,
        "alt": 35,
        "ast": 32,
        "has_nephropathy": True,
        "has_retinopathy": True,
        "has_neuropathy": True,
        "has_cardiovascular": True,
        "has_hypertension": True,
    },
    # High potassium - hyperkalemia risk
    {
        "name": "high_potassium",
        "diabetes_type": "type_2",
        "years_with_diabetes": 8,
        "age": 58,
        "hba1c": 7.8,
        "fasting_glucose": 160,
        "egfr": 55,
        "creatinine": 1.4,
        "potassium": 5.8,
        "alt": 28,
        "ast": 26,
        "has_nephropathy": True,
        "has_retinopathy": False,
        "has_neuropathy": False,
        "has_cardiovascular": True,
        "has_hypertension": True,
    },
    # Elderly, frail - hypoglycemia risk
    {
        "name": "elderly_frail",
        "diabetes_type": "type_2",
        "years_with_diabetes": 20,
        "age": 82,
        "hba1c": 7.0,
        "fasting_glucose": 130,
        "egfr": 48,
        "creatinine": 1.5,
        "potassium": 4.5,
        "alt": 22,
        "ast": 20,
        "has_nephropathy": True,
        "has_retinopathy": True,
        "has_neuropathy": True,
        "has_cardiovascular": True,
        "has_hypertension": True,
    },
    # Type 1, well controlled
    {
        "name": "type1_controlled",
        "diabetes_type": "type_1",
        "years_with_diabetes": 15,
        "age": 35,
        "hba1c": 6.5,
        "fasting_glucose": 110,
        "egfr": 95,
        "creatinine": 0.8,
        "potassium": 4.0,
        "alt": 20,
        "ast": 18,
        "has_nephropathy": False,
        "has_retinopathy": False,
        "has_neuropathy": False,
        "has_cardiovascular": False,
        "has_hypertension": False,
    },
    # Liver disease
    {
        "name": "liver_disease",
        "diabetes_type": "type_2",
        "years_with_diabetes": 12,
        "age": 55,
        "hba1c": 7.2,
        "fasting_glucose": 140,
        "egfr": 70,
        "creatinine": 1.1,
        "potassium": 4.3,
        "alt": 95,
        "ast": 88,
        "has_nephropathy": False,
        "has_retinopathy": False,
        "has_neuropathy": False,
        "has_cardiovascular": False,
        "has_hypertension": False,
    },
    # Cardiovascular high risk
    {
        "name": "cv_high_risk",
        "diabetes_type": "type_2",
        "years_with_diabetes": 18,
        "age": 68,
        "hba1c": 8.0,
        "fasting_glucose": 170,
        "egfr": 50,
        "creatinine": 1.5,
        "potassium": 4.6,
        "alt": 30,
        "ast": 28,
        "has_nephropathy": True,
        "has_retinopathy": True,
        "has_neuropathy": True,
        "has_cardiovascular": True,
        "has_hypertension": True,
    },
]


# ============================================================================
# STEP 2: Generate Labels Using Rule Engine (Weak Supervision)
# ============================================================================


def generate_training_data():
    """Generate training data using rule engine as the label source."""
    logger.info("Generating training data from rule engine...")

    rules = DiabeticDrugRules()

    data = []

    for drug in TRAINING_DRUGS:
        for profile in PATIENT_PROFILES:
            # Get rule-based assessment
            assessment = rules.assess_drug_risk(
                drug_name=drug,
                patient=profile,
                current_medications=[],  # Simplified - no interaction context
            )

            # Create training sample
            sample = {
                "drug_name": drug,
                # Patient features
                "egfr": profile["egfr"],
                "potassium": profile["potassium"],
                "hba1c": profile["hba1c"],
                "alt": profile["alt"],
                "age": profile["age"],
                "has_nephropathy": 1 if profile["has_nephropathy"] else 0,
                "has_cardiovascular": 1 if profile["has_cardiovascular"] else 0,
                "has_neuropathy": 1 if profile["has_neuropathy"] else 0,
                "has_hypertension": 1 if profile["has_hypertension"] else 0,
                "diabetes_type_1": 1 if profile["diabetes_type"] == "type_1" else 0,
                # Label from rules
                "risk_level": assessment.risk_level,
                "risk_score": assessment.risk_score,
            }
            data.append(sample)

    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} training samples")
    logger.info(f"Label distribution:\n{df['risk_level'].value_counts()}")

    return df


# ============================================================================
# STEP 3: Feature Engineering with Drug Class Knowledge
# ============================================================================


def get_drug_class_features(drug_name: str, rules: DiabeticDrugRules) -> dict:
    """Extract drug class features from rules knowledge."""
    drug_lower = drug_name.lower()

    features = {
        "is_hypoglycemia_risk": 0,
        "is_hyperglycemia_risk": 0,
        "is_nephrotoxic": 0,
        "is_hyperkalemia_risk": 0,
        "is_hepatotoxic": 0,
        "is_cardioprotective": 0,
        "masks_hypoglycemia": 0,
        "is_corticosteroid": 0,
        "is_nsaid": 0,
        "is_ace_arb": 0,
        "is_diuretic": 0,
        "is_beta_blocker": 0,
        "is_sulfonylurea": 0,
        "is_sglt2": 0,
        "is_glp1": 0,
    }

    # Check hypoglycemia risk drugs
    for cat, drugs in rules.HYPOGLYCEMIA_RISK_DRUGS.items():
        if any(d.lower() in drug_lower or drug_lower in d.lower() for d in drugs):
            features["is_hypoglycemia_risk"] = 1
            break

    # Check hyperglycemia risk drugs
    for cat, drugs in rules.HYPERGLYCEMIA_RISK_DRUGS.items():
        if any(d.lower() in drug_lower or drug_lower in d.lower() for d in drugs):
            features["is_hyperglycemia_risk"] = 1
            # Also check specific class
            if cat == "corticosteroids":
                features["is_corticosteroid"] = 1
            break

    # Check nephrotoxic
    if any(
        d.lower() in drug_lower or drug_lower in d.lower()
        for d in rules.NEPHROTOXIC_DRUGS
    ):
        features["is_nephrotoxic"] = 1

    # Check hyperkalemia risk
    if any(
        d.lower() in drug_lower or drug_lower in d.lower()
        for d in rules.HYPERKALEMIA_RISK_DRUGS
    ):
        features["is_hyperkalemia_risk"] = 1

    # Check hepatotoxic
    if any(
        d.lower() in drug_lower or drug_lower in d.lower()
        for d in rules.HEPATOTOXIC_DRUGS
    ):
        features["is_hepatotoxic"] = 1

    # Check cardioprotective
    if any(
        d.lower() in drug_lower or drug_lower in d.lower()
        for d in rules.CARDIOPROTECTIVE_IN_DIABETES
    ):
        features["is_cardioprotective"] = 1

    # Check masks hypoglycemia
    if any(
        d.lower() in drug_lower or drug_lower in d.lower()
        for d in rules.MASK_HYPOGLYCEMIA
    ):
        features["masks_hypoglycemia"] = 1
        features["is_beta_blocker"] = 1

    # NSAIDs
    nsaids = [
        "ibuprofen",
        "naproxen",
        "diclofenac",
        "celecoxib",
        "meloxicam",
        "indomethacin",
        "ketorolac",
    ]
    if any(n in drug_lower for n in nsaids):
        features["is_nsaid"] = 1

    # ACE/ARB
    ace_arbs = ["pril", "sartan"]
    if any(a in drug_lower for a in ace_arbs):
        features["is_ace_arb"] = 1

    # Diuretics
    diuretics = [
        "furosemide",
        "hydrochlorothiazide",
        "chlorthalidone",
        "bumetanide",
        "torsemide",
        "metolazone",
    ]
    if any(d in drug_lower for d in diuretics):
        features["is_diuretic"] = 1

    # Sulfonylureas
    sulfonylureas = ["glipizide", "glyburide", "glimepiride", "gliclazide"]
    if any(s in drug_lower for s in sulfonylureas):
        features["is_sulfonylurea"] = 1

    # SGLT2 inhibitors
    sglt2 = [
        "gliflozin",
        "empagliflozin",
        "canagliflozin",
        "dapagliflozin",
        "ertugliflozin",
    ]
    if any(s in drug_lower for s in sglt2):
        features["is_sglt2"] = 1

    # GLP-1 agonists
    glp1 = ["glutide", "liraglutide", "semaglutide", "dulaglutide", "exenatide"]
    if any(g in drug_lower for g in glp1):
        features["is_glp1"] = 1

    return features


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix with patient context + drug class features."""
    logger.info("Preparing features...")

    rules = DiabeticDrugRules()

    # Patient numerical features
    patient_cols = ["egfr", "potassium", "hba1c", "alt", "age"]
    X_patient = df[patient_cols].values

    # Patient categorical features
    cat_cols = [
        "has_nephropathy",
        "has_cardiovascular",
        "has_neuropathy",
        "has_hypertension",
        "diabetes_type_1",
    ]
    X_cat = df[cat_cols].values

    # Drug class features
    drug_class_features = []
    for drug in df["drug_name"]:
        features = get_drug_class_features(drug, rules)
        drug_class_features.append(list(features.values()))
    X_drug_class = np.array(drug_class_features)

    # Drug name TF-IDF (character n-grams for name patterns)
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=500, min_df=1
    )
    X_tfidf = tfidf.fit_transform(df["drug_name"].str.lower())

    # Combine all features
    X = hstack(
        [csr_matrix(X_patient), csr_matrix(X_cat), csr_matrix(X_drug_class), X_tfidf]
    )

    # Labels
    risk_order = ["safe", "caution", "high_risk", "contraindicated", "fatal"]
    le = LabelEncoder()
    le.fit(risk_order)
    y = le.transform(df["risk_level"])

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Feature breakdown:")
    logger.info(f"  Patient numerical: {len(patient_cols)}")
    logger.info(f"  Patient categorical: {len(cat_cols)}")
    logger.info(f"  Drug class features: {X_drug_class.shape[1]}")
    logger.info(f"  Drug name TF-IDF: {X_tfidf.shape[1]}")

    return X, y, tfidf, le


# ============================================================================
# STEP 4: Train with Proper Class Balancing
# ============================================================================


def compute_class_weights(y):
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def train_xgboost(X_train, y_train, class_weights):
    """Train XGBoost with class weights."""
    logger.info("Training XGBoost...")

    try:
        from xgboost import XGBClassifier

        # Convert class weights to sample weights
        sample_weights = np.array([class_weights[c] for c in y_train])

        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )

        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model
    except ImportError:
        logger.warning("XGBoost not installed")
        return None


def train_random_forest(X_train, y_train, class_weights):
    """Train Random Forest with class weights."""
    logger.info("Training Random Forest...")

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, class_weights):
    """Train Gradient Boosting (fallback for LightGBM issues)."""
    logger.info("Training Gradient Boosting...")

    from sklearn.ensemble import GradientBoostingClassifier

    # Convert to dense if sparse
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    # Sample weights
    sample_weights = np.array([class_weights[c] for c in y_train])

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


# ============================================================================
# STEP 5: Evaluation with Focus on Safety Metrics
# ============================================================================


def evaluate_model(model, X_test, y_test, model_name: str, le):
    """Evaluate with focus on safety-critical metrics."""
    if model is None:
        return None

    logger.info(f"Evaluating {model_name}...")

    # Handle sparse/dense
    X_eval = X_test
    if hasattr(X_test, "toarray") and model_name == "Gradient Boosting":
        X_eval = X_test.toarray()

    y_pred = model.predict(X_eval)

    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Per-class metrics (focus on high-risk classes)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )

    # Safety-critical: recall for high_risk, contraindicated, fatal
    high_risk_recall = report.get("high_risk", {}).get("recall", 0)
    contra_recall = report.get("contraindicated", {}).get("recall", 0)
    fatal_recall = report.get("fatal", {}).get("recall", 0)

    # Weighted F1
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "weighted_f1": f1,
        "high_risk_recall": high_risk_recall,
        "contraindicated_recall": contra_recall,
        "fatal_recall": fatal_recall,
        "confusion_matrix": cm.tolist(),
    }

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Weighted F1: {f1:.4f}")
    logger.info(f"  High Risk Recall: {high_risk_recall:.4f}")
    logger.info(f"  Contraindicated Recall: {contra_recall:.4f}")
    logger.info(f"  Fatal Recall: {fatal_recall:.4f}")

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return metrics


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("  DrugGuard ML Training V2 - CLINICAL LABELS")
    print("=" * 70)
    print()
    print("This model is trained on:")
    print("  - Labels from CLINICAL RULE ENGINE (not DDI frequency)")
    print("  - Patient context features (eGFR, HbA1c, complications)")
    print("  - Drug class knowledge (from rules.py)")
    print()

    # Generate training data
    df = generate_training_data()

    # Augment with variations (add noise to patient features)
    logger.info("Augmenting data with variations...")
    augmented = []
    for _, row in df.iterrows():
        for _ in range(5):  # 5 variations per sample
            new_row = row.copy()
            new_row["egfr"] += random.uniform(-10, 10)
            new_row["potassium"] += random.uniform(-0.3, 0.3)
            new_row["hba1c"] += random.uniform(-0.5, 0.5)
            new_row["age"] += random.randint(-5, 5)
            augmented.append(new_row)

    df_augmented = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)
    logger.info(f"Augmented to {len(df_augmented)} samples")

    # Prepare features
    X, y, tfidf, le = prepare_features(df_augmented)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Class weights
    class_weights = compute_class_weights(y_train)
    logger.info(f"Class weights: {class_weights}")

    # Train models
    models = {}
    models["XGBoost"] = train_xgboost(X_train, y_train, class_weights)
    models["Random Forest"] = train_random_forest(X_train, y_train, class_weights)
    models["Gradient Boosting"] = train_gradient_boosting(
        X_train.toarray() if hasattr(X_train, "toarray") else X_train,
        y_train,
        class_weights,
    )

    # Evaluate
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name, le)
        results.append(metrics)

    # Save best model
    best = max([r for r in results if r], key=lambda x: x["weighted_f1"])
    best_model = models[best["model_name"]]

    logger.info(f"\n🏆 Best Model: {best['model_name']}")

    # Save artifacts
    joblib.dump(best_model, MODELS_DIR / "diabetic_risk_model_v2.joblib")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_v2.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder_v2.joblib")

    # Save summary
    summary = {
        "training_date": datetime.now().isoformat(),
        "version": "2.0",
        "description": "Trained on rule engine labels with patient context",
        "best_model": best["model_name"],
        "accuracy": best["accuracy"],
        "weighted_f1": best["weighted_f1"],
        "high_risk_recall": best["high_risk_recall"],
        "contraindicated_recall": best["contraindicated_recall"],
        "fatal_recall": best["fatal_recall"],
        "training_samples": len(df_augmented),
        "label_distribution": df_augmented["risk_level"].value_counts().to_dict(),
    }

    with open(MODELS_DIR / "training_results_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {MODELS_DIR / 'diabetic_risk_model_v2.joblib'}")
    print(f"Results saved to: {MODELS_DIR / 'training_results_v2.json'}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
