"""
ML Model Training and Evaluation Script.

Trains XGBoost, Random Forest, and LightGBM on REAL TWOSIDES data
and outputs proper evaluation metrics.
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directories
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data(db_path: str, sample_size: int = 100000):
    """Load balanced sample from database."""
    logger.info(f"Loading {sample_size} samples from database...")
    
    conn = sqlite3.connect(db_path)
    
    # Get balanced sample across severity levels
    query = """
        SELECT drug1_name, drug2_name, severity
        FROM twosides_interactions
        WHERE severity IS NOT NULL AND drug1_name IS NOT NULL AND drug2_name IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(sample_size,))
    conn.close()
    
    logger.info(f"Loaded {len(df)} interactions")
    logger.info(f"Severity distribution:\n{df['severity'].value_counts()}")
    
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features from drug names."""
    logger.info("Preparing features...")
    
    # Combine drug names for text features
    df['drug_pair'] = df['drug1_name'].str.lower() + ' ' + df['drug2_name'].str.lower()
    
    # TF-IDF on drug name character n-grams
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(2, 4), 
        max_features=3000,
        min_df=2
    )
    X_char = tfidf_char.fit_transform(df['drug_pair'])
    
    # TF-IDF on drug name words
    tfidf_word = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2
    )
    X_word = tfidf_word.fit_transform(df['drug_pair'])
    
    # Combine features
    X = hstack([X_char, X_word])
    
    # Encode labels
    le = LabelEncoder()
    # Order: minor=0, moderate=1, major=2, fatal=3
    severity_order = ['minor', 'moderate', 'major', 'fatal']
    le.fit(severity_order)
    y = le.transform(df['severity'])
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Classes: {le.classes_}")
    
    return X, y, tfidf_char, tfidf_word, le


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    logger.info("Training XGBoost...")
    
    try:
        from xgboost import XGBClassifier
        
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    except ImportError:
        logger.warning("XGBoost not installed, skipping...")
        return None


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    logger.info("Training Random Forest...")
    
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    logger.info("Training LightGBM...")
    
    try:
        from lightgbm import LGBMClassifier
        
        model = LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Simpler fit without eval_set due to compatibility
        model.fit(X_train, y_train)
        
        return model
    except ImportError:
        logger.warning("LightGBM not installed, skipping...")
        return None
    except Exception as e:
        logger.warning(f"LightGBM training failed: {e}")
        return None


def evaluate_model(model, X_test, y_test, model_name: str, label_encoder):
    """Evaluate model and return metrics."""
    if model is None:
        return None
    
    logger.info(f"Evaluating {model_name}...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Multi-class ROC AUC
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  ROC AUC:   {auc:.4f}")
    
    return metrics


def save_results(results: list, models: dict, vectorizers: tuple, label_encoder):
    """Save models and results."""
    logger.info("Saving models and results...")
    
    # Save models
    for name, model in models.items():
        if model is not None:
            path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_model.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved: {path}")
    
    # Save vectorizers
    joblib.dump(vectorizers[0], MODELS_DIR / "tfidf_char.joblib")
    joblib.dump(vectorizers[1], MODELS_DIR / "tfidf_word.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    
    # Save results summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'models': []
    }
    
    for r in results:
        if r:
            summary['models'].append({
                'name': r['model_name'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1_score': r['f1_score'],
                'auc_roc': r['auc_roc']
            })
    
    with open(MODELS_DIR / "training_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {MODELS_DIR / 'training_results.json'}")


def print_final_summary(results: list):
    """Print final summary table."""
    print("\n" + "=" * 70)
    print("  FINAL MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 70)
    
    for r in results:
        if r:
            print(f"{r['model_name']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['auc_roc']:>10.4f}")
    
    print("=" * 70)
    
    # Best model
    valid_results = [r for r in results if r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['f1_score'])
        print(f"\n🏆 Best Model: {best['model_name']} (F1: {best['f1_score']:.4f})")
    
    print()


def main():
    """Main training function."""
    print("=" * 70)
    print("  DrugGuard ML Training - REAL DATA")
    print("=" * 70)
    print()
    
    # Database path
    db_path = Path(__file__).parent / "drug_interactions.db"
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    # Load data
    df = load_data(str(db_path), sample_size=100000)
    
    # Prepare features
    X, y, tfidf_char, tfidf_word, le = prepare_features(df)
    
    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train models
    models = {}
    models['XGBoost'] = train_xgboost(X_train, y_train, X_val, y_val)
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['LightGBM'] = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Evaluate all models
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name, le)
        results.append(metrics)
        
        if metrics:
            print(f"\n{name} Classification Report:")
            print(metrics['classification_report'])
    
    # Save everything
    save_results(results, models, (tfidf_char, tfidf_word), le)
    
    # Print summary
    print_final_summary(results)
    
    return results


if __name__ == "__main__":
    main()
