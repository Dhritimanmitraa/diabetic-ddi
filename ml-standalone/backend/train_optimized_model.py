"""
Optimized Drug-Drug Interaction ML Model - Version 3.0

Features:
1. SMOTE for class balancing
2. Optimized XGBoost (no ensemble to save memory)
3. Expanded drug knowledge base (200+ patterns)

Target: 93-94% accuracy (memory-efficient version)
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import sqlite3
import random
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import drug knowledge from train_ultimate_model
from train_ultimate_model import (
    get_drug_classes, 
    get_interaction_features,
    DRUG_CLASS_PATTERNS,
    SPECIFIC_DRUG_CLASSES,
    HIGH_RISK_INTERACTIONS
)


def train_optimized_model():
    """Train optimized model - memory efficient"""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from scipy.sparse import hstack, csr_matrix
    import joblib
    
    try:
        from imblearn.over_sampling import SMOTE
        use_smote = True
    except ImportError:
        use_smote = False
    
    try:
        from xgboost import XGBClassifier
        use_xgboost = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        use_xgboost = False
    
    # Load data
    db_path = Path("C:/Drug/backend/drug_interactions.db")
    if not db_path.exists():
        db_path = Path(__file__).parent / ".." / ".." / "backend" / "drug_interactions.db"
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT drug1_name, drug2_name, severity
        FROM twosides_interactions
        WHERE drug1_name IS NOT NULL AND drug2_name IS NOT NULL
        LIMIT 50000
    """)
    interactions = cursor.fetchall()
    conn.close()
    
    logger.info(f"Loaded {len(interactions)} interactions")
    
    # Prepare features
    X_text = []
    X_int_features = []
    y = []
    
    severity_map = {
        None: 1, '': 1, 'mild': 1, 'low': 1, 'minor': 1,
        'moderate': 2, 'medium': 2,
        'high': 3, 'severe': 3, 'major': 3,
        'contraindicated': 4, 'critical': 4
    }
    
    for drug1, drug2, severity in interactions:
        if not drug1 or not drug2:
            continue
        
        drugs = sorted([drug1.lower().strip(), drug2.lower().strip()])
        X_text.append(f"{drugs[0]} {drugs[1]}")
        
        classes1 = get_drug_classes(drugs[0])
        classes2 = get_drug_classes(drugs[1])
        features = get_interaction_features(classes1, classes2)
        X_int_features.append(list(features.values()))
        
        if severity is None:
            label = 1
        elif isinstance(severity, (int, float)):
            label = min(4, max(1, int(severity)))
        else:
            label = severity_map.get(str(severity).lower().strip(), 1)
        y.append(label)
    
    # Add negative samples
    all_drugs = list(set(
        d.lower().strip() for d1, d2, _ in interactions 
        for d in [d1, d2] if d
    ))
    
    random.seed(42)
    interaction_set = set(X_text)
    neg_count = min(len([l for l in y if l > 0]), 12000)  # Less negatives
    
    for _ in range(neg_count * 3):
        if len([l for l in y if l == 0]) >= neg_count:
            break
        d1, d2 = random.choice(all_drugs), random.choice(all_drugs)
        if d1 != d2:
            drugs = sorted([d1, d2])
            pair = f"{drugs[0]} {drugs[1]}"
            if pair not in interaction_set:
                X_text.append(pair)
                classes1, classes2 = get_drug_classes(drugs[0]), get_drug_classes(drugs[1])
                X_int_features.append(list(get_interaction_features(classes1, classes2).values()))
                y.append(0)
                interaction_set.add(pair)
    
    logger.info(f"Total samples: {len(X_text)}")
    
    # Create feature matrices (smaller)
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=4000, min_df=3)
    X_tfidf = tfidf.fit_transform(X_text)
    
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=1500, min_df=3)
    X_word = word_vec.fit_transform(X_text)
    
    X_int = csr_matrix(np.array(X_int_features))
    X = hstack([X_tfidf, X_word, X_int])
    y = np.array(y)
    
    logger.info(f"Feature matrix: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE (on subset to save memory)
    if use_smote:
        logger.info("Applying SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        # Use sparse-friendly approach
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train.toarray(), y_train)
        X_train_resampled = csr_matrix(X_train_resampled)
        logger.info(f"After SMOTE: {len(y_train_resampled)} samples")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Train optimized XGBoost
    logger.info("Training XGBoost with optimized parameters...")
    
    if use_xgboost:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.12,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200, max_depth=8, random_state=42)
    
    model.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"OPTIMIZED MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"WEIGHTED F1 SCORE: {f1:.4f}")
    logger.info(f"{'='*60}\n")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[
        'none', 'mild', 'moderate', 'severe', 'contraindicated'
    ][:len(set(y))]))
    
    # Save
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(model, models_dir / "drug_interaction_model.joblib")
    joblib.dump(tfidf, models_dir / "drug_vectorizer.joblib")
    joblib.dump(word_vec, models_dir / "word_vectorizer.joblib")
    
    sample_features = get_interaction_features([], [])
    
    model_info = {
        "version": "3.0.0",
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "training_date": datetime.now().isoformat(),
        "num_training_samples": len(X_text),
        "num_training_after_smote": len(y_train_resampled),
        "feature_count": X.shape[1],
        "model_type": "XGBoost-Optimized",
        "classes": ["none", "mild", "moderate", "severe", "contraindicated"][:len(set(y))],
        "interaction_feature_names": list(sample_features.keys()),
        "enhancements": [
            "SMOTE class balancing",
            "200+ drug class patterns",
            "High-risk interaction detection",
            "Pharmacological class features"
        ]
    }
    
    with open(models_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"Model saved to: {models_dir}")
    logger.info(f"Final accuracy: {accuracy:.4f}")
    
    return True


if __name__ == "__main__":
    success = train_optimized_model()
    sys.exit(0 if success else 1)
