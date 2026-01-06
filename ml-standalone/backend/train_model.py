"""
Train Drug-Drug Interaction ML Model

This script trains an XGBoost classifier on the TWOSIDES database
to predict drug interactions and their severity.

Usage:
    python train_model.py
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))


def train_model():
    """Main training function"""
    try:
        import numpy as np
        import sqlite3
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        import joblib
        
        # Try to import XGBoost, fall back to RandomForest if not available
        try:
            from xgboost import XGBClassifier
            use_xgboost = True
            logger.info("Using XGBoost classifier")
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            use_xgboost = False
            logger.info("XGBoost not available, using RandomForest")
        
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install scikit-learn xgboost joblib numpy pandas")
        return False
    
    # Database paths to try
    db_paths = [
        Path(__file__).parent / ".." / ".." / "backend" / "drug_interactions.db",
        Path(__file__).parent / ".." / ".." / ".." / "backend" / "drug_interactions.db",
        Path("C:/Drug/backend/drug_interactions.db"),
    ]
    
    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break
    
    if not db_path:
        logger.warning("TWOSIDES database not found. Creating synthetic training data...")
        return train_with_synthetic_data()
    
    logger.info(f"Loading data from: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check available tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Available tables: {tables}")
    
    # Load interactions data
    try:
        if 'twosides_interactions' in tables:
            cursor.execute("""
                SELECT drug1_name, drug2_name, severity
                FROM twosides_interactions
                WHERE drug1_name IS NOT NULL AND drug2_name IS NOT NULL
                LIMIT 50000
            """)
        elif 'interactions' in tables:
            cursor.execute("""
                SELECT drug1, drug2, severity
                FROM interactions
                LIMIT 50000
            """)
        else:
            logger.warning("No interaction table found. Using synthetic data.")
            conn.close()
            return train_with_synthetic_data()
        
        interactions = cursor.fetchall()
        logger.info(f"Loaded {len(interactions)} interactions from database")
        
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        conn.close()
        return train_with_synthetic_data()
    
    conn.close()
    
    if len(interactions) < 100:
        logger.warning("Not enough data in database. Using synthetic data.")
        return train_with_synthetic_data()
    
    # Prepare training data
    logger.info("Preparing training data...")
    
    # Create feature strings (alphabetically sorted drug names)
    X_text = []
    y = []
    
    severity_to_label = {
        None: 1,
        '': 1,
        'mild': 1,
        'low': 1,
        'minor': 1,
        'moderate': 2,
        'medium': 2,
        'high': 3,
        'severe': 3,
        'major': 3,
        'contraindicated': 4,
        'critical': 4
    }
    
    for drug1, drug2, severity in interactions:
        if not drug1 or not drug2:
            continue
            
        drugs = sorted([drug1.lower().strip(), drug2.lower().strip()])
        X_text.append(f"{drugs[0]} {drugs[1]}")
        
        # Map severity to numeric label
        if severity is None:
            label = 1  # Default to mild for recorded interactions
        elif isinstance(severity, (int, float)):
            label = min(4, max(1, int(severity)))
        else:
            label = severity_to_label.get(str(severity).lower().strip(), 1)
        
        y.append(label)
    
    # Add negative samples (no interaction)
    # Get unique drugs
    all_drugs = set()
    for drug1, drug2, _ in interactions:
        if drug1:
            all_drugs.add(drug1.lower().strip())
        if drug2:
            all_drugs.add(drug2.lower().strip())
    
    drug_list = list(all_drugs)
    logger.info(f"Unique drugs: {len(drug_list)}")
    
    # Create negative samples (random pairs assumed to have no interaction)
    import random
    random.seed(42)
    
    interaction_set = set(X_text)
    negative_count = min(len(X_text), 10000)
    
    for _ in range(negative_count * 3):  # Try more times to get enough
        if len([x for x in y if x == 0]) >= negative_count:
            break
            
        d1 = random.choice(drug_list)
        d2 = random.choice(drug_list)
        if d1 != d2:
            drugs = sorted([d1, d2])
            pair = f"{drugs[0]} {drugs[1]}"
            if pair not in interaction_set:
                X_text.append(pair)
                y.append(0)  # No interaction
                interaction_set.add(pair)
    
    logger.info(f"Total samples: {len(X_text)} (positive: {sum(1 for l in y if l > 0)}, negative: {sum(1 for l in y if l == 0)})")
    
    # Vectorize text
    logger.info("Vectorizing drug names...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=5000,
        min_df=2
    )
    X = vectorizer.fit_transform(X_text)
    y = np.array(y)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    logger.info("Training model...")
    
    if use_xgboost:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"\nModel Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[
        'none', 'mild', 'moderate', 'severe', 'contraindicated'
    ][:len(set(y))]))
    
    # Save model
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    logger.info(f"Saving model to {models_dir}...")
    
    joblib.dump(model, models_dir / "drug_interaction_model.joblib")
    joblib.dump(vectorizer, models_dir / "drug_vectorizer.joblib")
    
    # Save model info
    model_info = {
        "version": "1.0.0",
        "accuracy": float(accuracy),
        "training_date": datetime.now().isoformat(),
        "num_training_samples": len(X_text),
        "feature_count": X.shape[1],
        "model_type": "XGBoost" if use_xgboost else "RandomForest",
        "classes": ["none", "mild", "moderate", "severe", "contraindicated"][:len(set(y))]
    }
    
    with open(models_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Model saved to: {models_dir}")
    
    return True


def train_with_synthetic_data():
    """Train model with synthetic data when database is not available"""
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        import joblib
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        return False
    
    logger.info("Generating synthetic training data...")
    
    # Common drugs for synthetic data
    drugs = [
        "aspirin", "warfarin", "metformin", "lisinopril", "atorvastatin",
        "omeprazole", "amlodipine", "metoprolol", "losartan", "gabapentin",
        "hydrocodone", "acetaminophen", "ibuprofen", "prednisone", "tramadol",
        "albuterol", "fluticasone", "montelukast", "sertraline", "escitalopram",
        "alprazolam", "lorazepam", "zolpidem", "trazodone", "quetiapine",
        "aripiprazole", "risperidone", "lithium", "valproic acid", "carbamazepine",
        "phenytoin", "levetiracetam", "topiramate", "lamotrigine", "clonazepam",
        "clopidogrel", "rivaroxaban", "apixaban", "enoxaparin", "heparin",
        "furosemide", "hydrochlorothiazide", "spironolactone", "potassium", "magnesium"
    ]
    
    # Known interactions (simplified)
    known_interactions = [
        ("aspirin", "warfarin", 3),      # Severe - bleeding risk
        ("ibuprofen", "warfarin", 3),    # Severe - bleeding risk  
        ("metformin", "alcohol", 2),     # Moderate - lactic acidosis
        ("lisinopril", "potassium", 2),  # Moderate - hyperkalemia
        ("sertraline", "tramadol", 3),   # Severe - serotonin syndrome
        ("warfarin", "vitamin k", 2),    # Moderate - reduced effect
        ("clopidogrel", "omeprazole", 2),# Moderate - reduced effect
        ("lithium", "ibuprofen", 3),     # Severe - lithium toxicity
        ("methotrexate", "nsaids", 3),   # Severe - toxicity
        ("simvastatin", "grapefruit", 2),# Moderate - increased levels
    ]
    
    import random
    random.seed(42)
    
    X_text = []
    y = []
    
    # Add known interactions
    for d1, d2, severity in known_interactions:
        drugs_sorted = sorted([d1.lower(), d2.lower()])
        X_text.append(f"{drugs_sorted[0]} {drugs_sorted[1]}")
        y.append(severity)
    
    # Generate more positive samples (interactions with varying severity)
    for _ in range(500):
        d1, d2 = random.sample(drugs, 2)
        drugs_sorted = sorted([d1.lower(), d2.lower()])
        pair = f"{drugs_sorted[0]} {drugs_sorted[1]}"
        if pair not in X_text:
            X_text.append(pair)
            y.append(random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0])
    
    # Generate negative samples (no interaction)
    for _ in range(700):
        d1, d2 = random.sample(drugs, 2)
        drugs_sorted = sorted([d1.lower(), d2.lower()])
        pair = f"{drugs_sorted[0]} {drugs_sorted[1]}"
        if pair not in X_text:
            X_text.append(pair)
            y.append(0)
    
    logger.info(f"Generated {len(X_text)} synthetic samples")
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        max_features=3000
    )
    X = vectorizer.fit_transform(X_text)
    y = np.array(y)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    
    # Save
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, models_dir / "drug_interaction_model.joblib")
    joblib.dump(vectorizer, models_dir / "drug_vectorizer.joblib")
    
    model_info = {
        "version": "1.0.0-synthetic",
        "accuracy": float(accuracy),
        "training_date": datetime.now().isoformat(),
        "num_training_samples": len(X_text),
        "feature_count": X.shape[1],
        "model_type": "RandomForest",
        "data_source": "synthetic",
        "classes": ["none", "mild", "moderate", "severe"]
    }
    
    with open(models_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Model saved to: {models_dir}")
    return True


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
