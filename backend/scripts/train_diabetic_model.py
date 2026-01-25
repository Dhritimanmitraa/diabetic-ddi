"""
Train Diabetic Risk ML Model.

Creates a properly formatted model artifact for the DiabeticMLPredictor.
Uses synthetic training data based on clinical rules to bootstrap the model.
"""
import os
import sys
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Risk levels
RISK_LEVELS = ["safe", "caution", "high_risk", "contraindicated", "fatal"]
RISK_TO_INT = {r: i for i, r in enumerate(RISK_LEVELS)}
INT_TO_RISK = {i: r for i, r in enumerate(RISK_LEVELS)}

# Drug categories for synthetic data generation
HIGH_RISK_DRUGS = [
    "metformin", "glipizide", "glimepiride", "glyburide", "pioglitazone",
    "lisinopril", "enalapril", "losartan", "spironolactone", "amiloride",
    "ibuprofen", "naproxen", "diclofenac", "celecoxib", "aspirin",
    "prednisone", "dexamethasone", "hydrocortisone",
    "warfarin", "digoxin", "lithium", "phenytoin"
]

CAUTION_DRUGS = [
    "atorvastatin", "simvastatin", "omeprazole", "pantoprazole",
    "amlodipine", "metoprolol", "carvedilol", "furosemide",
    "gabapentin", "pregabalin", "tramadol", "acetaminophen",
    "sertraline", "fluoxetine", "duloxetine", "amitriptyline"
]

SAFE_DRUGS = [
    "vitamin_d", "calcium", "multivitamin", "fish_oil", "probiotics",
    "melatonin", "magnesium", "vitamin_b12", "folic_acid", "zinc"
]

FATAL_COMBINATIONS = [
    ("trimethoprim", {"has_nephropathy": True, "potassium": 5.5}),
    ("potassium_chloride", {"potassium": 5.8}),
    ("spironolactone", {"potassium": 5.5, "has_nephropathy": True}),
]


def hash_text(text: str, n_features: int) -> np.ndarray:
    """Hash text into a fixed-size vector."""
    vec = np.zeros(n_features, dtype=np.float32)
    if not isinstance(text, str):
        return vec
    lower = text.lower()
    for idx, ch in enumerate(lower):
        bucket = (hash(ch + str(idx)) % n_features)
        vec[bucket] += 1.0
    return vec


def generate_patient():
    """Generate a random patient profile."""
    age = np.random.randint(25, 85)
    gender = np.random.choice(["male", "female"])
    
    # Lab values with some abnormals
    egfr = np.random.choice([
        np.random.uniform(90, 120),  # Normal
        np.random.uniform(60, 89),   # Mild decrease
        np.random.uniform(30, 59),   # Moderate decrease
        np.random.uniform(15, 29),   # Severe decrease
    ], p=[0.4, 0.3, 0.2, 0.1])
    
    creatinine = max(0.6, 120 / egfr * 0.8 + np.random.uniform(-0.2, 0.5))
    potassium = np.random.choice([
        np.random.uniform(3.5, 5.0),  # Normal
        np.random.uniform(5.0, 5.5),  # Slightly elevated
        np.random.uniform(5.5, 6.5),  # High (dangerous)
    ], p=[0.7, 0.2, 0.1])
    
    fasting_glucose = np.random.choice([
        np.random.uniform(70, 100),   # Normal
        np.random.uniform(100, 126),  # Prediabetes
        np.random.uniform(126, 250),  # Diabetes
        np.random.uniform(250, 400),  # Uncontrolled
    ], p=[0.1, 0.2, 0.5, 0.2])
    
    return {
        "age": age,
        "gender": gender,
        "creatinine": creatinine,
        "potassium": potassium,
        "fasting_glucose": fasting_glucose,
        "egfr": egfr,
        "has_nephropathy": egfr < 60 or np.random.random() < 0.2,
        "has_retinopathy": np.random.random() < 0.15,
        "has_neuropathy": np.random.random() < 0.2,
        "has_cardiovascular": np.random.random() < 0.25,
        "has_hypertension": np.random.random() < 0.4,
        "has_hyperlipidemia": np.random.random() < 0.35,
        "has_obesity": np.random.random() < 0.3,
    }


def determine_risk_level(drug: str, patient: dict) -> str:
    """Determine risk level based on clinical rules."""
    drug_lower = drug.lower().replace("_", " ").replace("-", " ")
    
    # Fatal conditions
    if patient.get("potassium", 4.0) >= 5.8:
        if any(d in drug_lower for d in ["potassium", "spironolactone", "amiloride", "trimethoprim"]):
            return "fatal"
    
    if patient.get("egfr", 90) < 15:
        if any(d in drug_lower for d in ["metformin", "nsaid", "ibuprofen", "naproxen"]):
            return "fatal"
    
    # Contraindicated conditions
    if patient.get("potassium", 4.0) >= 5.5:
        if any(d in drug_lower for d in ["ace", "arb", "lisinopril", "enalapril", "losartan", "spironolactone"]):
            return "contraindicated"
    
    if patient.get("egfr", 90) < 30:
        if "metformin" in drug_lower:
            return "contraindicated"
        if any(d in drug_lower for d in ["nsaid", "ibuprofen", "naproxen", "diclofenac"]):
            return "contraindicated"
    
    # High risk conditions
    if patient.get("has_nephropathy") and patient.get("egfr", 90) < 45:
        if any(d in drug_lower for d in ["metformin", "nsaid", "contrast", "aminoglycoside"]):
            return "high_risk"
    
    if patient.get("potassium", 4.0) >= 5.0:
        if any(d in drug_lower for d in ["ace", "arb", "spironolactone", "trimethoprim"]):
            return "high_risk"
    
    # Drug-specific risks
    if drug_lower in [d.lower() for d in HIGH_RISK_DRUGS]:
        if patient.get("has_nephropathy") or patient.get("egfr", 90) < 60:
            return "high_risk"
        return "caution"
    
    if drug_lower in [d.lower() for d in CAUTION_DRUGS]:
        return "caution"
    
    if drug_lower in [d.lower() for d in SAFE_DRUGS]:
        return "safe"
    
    # Default based on patient condition
    if patient.get("egfr", 90) < 45 or patient.get("potassium", 4.0) >= 5.3:
        return "caution"
    
    return "safe"


def build_feature_vector(patient: dict, drug_name: str, hash_size: int) -> np.ndarray:
    """Build feature vector for a patient-drug pair."""
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
    return np.concatenate([np.array(num_features, dtype=np.float32), drug_vec])


def generate_training_data(n_samples: int = 5000, hash_size: int = 48):
    """Generate synthetic training data."""
    print(f"Generating {n_samples} training samples...")
    
    all_drugs = HIGH_RISK_DRUGS + CAUTION_DRUGS + SAFE_DRUGS + [
        "unknown_drug_1", "new_medication", "experimental_rx"
    ]
    
    X = []
    y = []
    
    for i in range(n_samples):
        patient = generate_patient()
        drug = np.random.choice(all_drugs)
        
        # Get risk level
        risk = determine_risk_level(drug, patient)
        
        # Build feature vector
        features = build_feature_vector(patient, drug, hash_size)
        
        X.append(features)
        y.append(RISK_TO_INT[risk])
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples} samples...")
    
    return np.array(X), np.array(y)


def train_model(output_dir: str = "./models"):
    """Train and save the ML model."""
    print("="*60)
    print("Diabetic Risk ML Model Training")
    print("="*60)
    
    hash_size = 48
    
    # Generate training data
    X, y = generate_training_data(n_samples=8000, hash_size=hash_size)
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {INT_TO_RISK[u]}: {c} ({100*c/len(y):.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numeric features (first 13)
    scaler = StandardScaler()
    X_train[:, :13] = scaler.fit_transform(X_train[:, :13])
    X_test[:, :13] = scaler.transform(X_test[:, :13])
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train XGBoost classifier
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        objective='multi:softprob',
        num_class=len(RISK_LEVELS),
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=RISK_LEVELS,
        zero_division=0
    ))
    
    # Create model artifact in expected dictionary format
    model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    artifact = {
        "model": model,
        "scaler": scaler,
        "hash_size": hash_size,
        "risk_to_int": RISK_TO_INT,
        "int_to_risk": INT_TO_RISK,
        "model_version": model_version,
        "accuracy": accuracy,
        "trained_at": datetime.now().isoformat(),
    }
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "diabetic_risk_model.pkl")
    joblib.dump(artifact, output_path)
    
    print(f"\n✅ Model saved to: {output_path}")
    print(f"   Version: {model_version}")
    print(f"   Accuracy: {accuracy:.4f}")
    
    return output_path, accuracy


if __name__ == "__main__":
    train_model("./models")
