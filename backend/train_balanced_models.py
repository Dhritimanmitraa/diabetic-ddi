"""
ML Model Training with BALANCED Classes.

Uses stratified sampling to ensure equal representation of all severity levels.
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
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack, csr_matrix
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_balanced_data(db_path: str, samples_per_class: int = 5000):
    """Load BALANCED sample with equal representation per severity class."""
    logger.info(f"Loading {samples_per_class} samples PER CLASS from database...")
    
    conn = sqlite3.connect(db_path)
    
    # For each severity, get equal samples
    dfs = []
    for severity in ['minor', 'moderate', 'major', 'fatal']:
        query = f"""
            SELECT drug1_name, drug2_name, severity
            FROM twosides_interactions
            WHERE severity = ?
            AND drug1_name IS NOT NULL AND drug2_name IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(severity, samples_per_class))
        dfs.append(df)
        logger.info(f"  {severity}: {len(df)} samples")
    
    conn.close()
    
    # Combine all
    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    logger.info(f"Total: {len(df)} samples (balanced)")
    logger.info(f"Final distribution:\n{df['severity'].value_counts()}")
    
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features from drug names."""
    logger.info("Preparing features...")
    
    df['drug_pair'] = df['drug1_name'].str.lower() + ' ' + df['drug2_name'].str.lower()
    
    # Character n-grams
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(2, 4), 
        max_features=3000,
        min_df=2
    )
    X_char = tfidf_char.fit_transform(df['drug_pair'])
    
    # Word n-grams
    tfidf_word = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2
    )
    X_word = tfidf_word.fit_transform(df['drug_pair'])
    
    X = hstack([X_char, X_word])
    
    # Encode labels with explicit ordering
    severity_order = ['minor', 'moderate', 'major', 'fatal']
    le = LabelEncoder()
    le.fit(severity_order)
    y = le.transform(df['severity'])
    
    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Classes: {le.classes_} -> [0, 1, 2, 3]")
    
    return X, y, tfidf_char, tfidf_word, le


def train_xgboost(X_train, y_train, X_val, y_val, class_weights):
    """Train XGBoost with class weights."""
    logger.info("Training XGBoost (with class weights)...")
    
    from xgboost import XGBClassifier
    
    # Convert class weights to sample weights
    sample_weights = np.array([class_weights[c] for c in y_train])
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights, 
              eval_set=[(X_val, y_val)], verbose=False)
    
    return model


def train_random_forest(X_train, y_train, class_weights):
    """Train Random Forest with class weights."""
    logger.info("Training Random Forest (with class weights)...")
    
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=3,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting as LightGBM alternative."""
    logger.info("Training Gradient Boosting...")
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Use smaller subset due to GradientBoosting's speed
    if X_train.shape[0] > 10000:
        indices = np.random.choice(X_train.shape[0], 10000, replace=False)
        X_sub = X_train[indices]
        y_sub = y_train[indices]
    else:
        X_sub, y_sub = X_train, y_train
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_sub.toarray() if hasattr(X_sub, 'toarray') else X_sub, y_sub)
    return model


def evaluate_model(model, X_test, y_test, model_name: str, label_encoder):
    """Evaluate model and return metrics."""
    if model is None:
        return None
    
    logger.info(f"Evaluating {model_name}...")
    
    X_eval = X_test.toarray() if hasattr(X_test, 'toarray') and 'Gradient' in model_name else X_test
    
    y_pred = model.predict(X_eval)
    
    try:
        y_proba = model.predict_proba(X_eval)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
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


def save_results(results, models, vectorizers, label_encoder):
    """Save models and results."""
    logger.info("Saving models and results...")
    
    for name, model in models.items():
        if model is not None:
            path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_balanced.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved: {path}")
    
    joblib.dump(vectorizers[0], MODELS_DIR / "tfidf_char_balanced.joblib")
    joblib.dump(vectorizers[1], MODELS_DIR / "tfidf_word_balanced.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder_balanced.joblib")
    
    summary = {
        'training_date': datetime.now().isoformat(),
        'balanced_training': True,
        'samples_per_class': 5000,
        'models': [
            {
                'name': r['model_name'],
                'accuracy': r['accuracy'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1_score': r['f1_score'],
                'auc_roc': r['auc_roc']
            }
            for r in results if r
        ]
    }
    
    with open(MODELS_DIR / "training_results_balanced.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {MODELS_DIR / 'training_results_balanced.json'}")


def print_final_summary(results):
    """Print final summary."""
    print("\n" + "=" * 80)
    print("  FINAL MODEL EVALUATION RESULTS (BALANCED TRAINING)")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 80)
    
    for r in results:
        if r:
            print(f"{r['model_name']:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['auc_roc']:>10.4f}")
    
    print("=" * 80)
    
    valid = [r for r in results if r]
    if valid:
        best = max(valid, key=lambda x: x['f1_score'])
        print(f"\n🏆 Best Model: {best['model_name']} (F1: {best['f1_score']:.4f})")
    
    print()


def main():
    """Main training function."""
    print("=" * 80)
    print("  DrugGuard ML Training - BALANCED CLASSES")
    print("=" * 80)
    print()
    
    db_path = Path(__file__).parent / "drug_interactions.db"
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    # Load BALANCED data
    df = load_balanced_data(str(db_path), samples_per_class=5000)
    
    # Prepare features
    X, y, tfidf_char, tfidf_word, le = prepare_features(df)
    
    # Compute class weights
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weights}")
    
    # Split: 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train models
    models = {}
    models['XGBoost'] = train_xgboost(X_train, y_train, X_val, y_val, class_weights)
    models['Random Forest'] = train_random_forest(X_train, y_train, class_weights)
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    
    # Evaluate
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name, le)
        results.append(metrics)
        
        if metrics:
            print(f"\n{name} Classification Report:")
            print(metrics['classification_report'])
    
    # Save
    save_results(results, models, (tfidf_char, tfidf_word), le)
    
    # Summary
    print_final_summary(results)
    
    return results


if __name__ == "__main__":
    main()
