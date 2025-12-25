"""
Find optimal classification threshold for imbalanced data.

This script:
1. Loads test data with balanced sampling
2. Finds optimal threshold using G-mean and Youden's J
3. Shows comparison table at different thresholds
4. Saves optimal threshold to models/optimal_threshold.json
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent if '__file__' in dir() else Path('.')
sys.path.insert(0, str(SCRIPT_DIR.parent))

from sklearn.metrics import (  # noqa: E402
    confusion_matrix, accuracy_score, f1_score, 
    roc_auc_score, roc_curve, precision_score, recall_score
)
import joblib  # noqa: E402
import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_features_fast(df: pd.DataFrame) -> tuple:
    """Fast feature preparation using vectorized operations."""
    n_samples = len(df)
    n_features = 2 + 100 + 60 + 80
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    
    X[:, 0] = df['drug1_matched'].fillna(0).astype(np.float32).values
    X[:, 1] = df['drug2_matched'].fillna(0).astype(np.float32).values
    
    def text_hash_vectorized(series, n_features, offset):
        for idx, val in enumerate(series):
            if pd.notna(val) and val:
                s = str(val).lower()
                for j, char in enumerate(s[:20]):
                    X[idx, offset + hash(char + str(j)) % n_features] += 1
    
    col_offset = 2
    text_hash_vectorized(df['drug1_class'].fillna(''), 50, col_offset)
    col_offset += 50
    text_hash_vectorized(df['drug2_class'].fillna(''), 50, col_offset)
    col_offset += 50
    text_hash_vectorized(df['drug1_mechanism'].fillna(''), 30, col_offset)
    col_offset += 30
    text_hash_vectorized(df['drug2_mechanism'].fillna(''), 30, col_offset)
    col_offset += 30
    text_hash_vectorized(df['drug1_name'].fillna(''), 40, col_offset)
    col_offset += 40
    text_hash_vectorized(df['drug2_name'].fillna(''), 40, col_offset)
    
    y = df['has_interaction'].values.astype(np.int32)
    return X, y


def find_optimal_threshold_gmean(y_true, y_proba):
    """Find optimal threshold using G-mean (sqrt(TPR * (1-FPR)))."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    return thresholds[ix], gmeans[ix]


def find_optimal_threshold_youden(y_true, y_proba):
    """Find optimal threshold using Youden's J (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    ix = np.argmax(j_scores)
    return thresholds[ix], j_scores[ix]


def load_models(model_dir: Path):
    """Load all trained models."""
    models = {}
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        model_path = model_dir / f"{model_name}_model.pkl"
        if model_path.exists():
            data = joblib.load(model_path)
            if isinstance(data, dict) and 'model' in data:
                models[model_name] = data.get('calibrated_model') or data.get('model')
            else:
                models[model_name] = data
            logger.info(f"Loaded {model_name}")
    return models


def main():
    print("\n" + "="*80)
    print("  OPTIMAL THRESHOLD FINDER")
    print("="*80)
    
    # Paths
    base_dir = Path(__file__).parent.parent if '__file__' in dir() else Path('.')
    data_dir = base_dir / "data" / "training"
    model_dir = base_dir / "models"
    test_path = data_dir / "test.csv"
    
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        return
    
    # Load balanced sample from test data
    print("\n1. Loading balanced test sample...")
    pos_samples = []
    neg_samples = []
    target_per_class = 10000
    
    for chunk in pd.read_csv(test_path, chunksize=100000, encoding='utf-8'):
        pos_chunk = chunk[chunk['has_interaction'] == 1]
        neg_chunk = chunk[chunk['has_interaction'] == 0]
        
        pos_count = sum(len(df) for df in pos_samples)
        neg_count = sum(len(df) for df in neg_samples)
        
        if pos_count < target_per_class:
            needed = target_per_class - pos_count
            pos_samples.append(pos_chunk.head(needed))
        
        if neg_count < target_per_class:
            needed = target_per_class - neg_count
            neg_samples.append(neg_chunk.head(needed))
        
        pos_count = sum(len(df) for df in pos_samples)
        neg_count = sum(len(df) for df in neg_samples)
        
        if pos_count >= target_per_class and neg_count >= target_per_class:
            break
    
    test_df = pd.concat(pos_samples + neg_samples, ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_test, y_test = prepare_features_fast(test_df)
    n_pos = np.sum(y_test == 1)
    n_neg = np.sum(y_test == 0)
    print(f"   Loaded {len(test_df)} samples: {n_pos} positive, {n_neg} negative")
    
    # Load models
    print("\n2. Loading models...")
    models = load_models(model_dir)
    if not models:
        logger.error("No models loaded!")
        return
    
    # Get ensemble probabilities
    print("\n3. Computing ensemble predictions...")
    probas = []
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        probas.append(proba)
        print(f"   {name}: mean proba = {np.mean(proba):.4f}")
    
    avg_proba = np.mean(probas, axis=0)
    print(f"   Ensemble: mean proba = {np.mean(avg_proba):.4f}")
    
    # Find optimal thresholds
    print("\n4. Finding optimal thresholds...")
    opt_gmean, score_gmean = find_optimal_threshold_gmean(y_test, avg_proba)
    opt_youden, score_youden = find_optimal_threshold_youden(y_test, avg_proba)
    
    print(f"\n   G-MEAN OPTIMAL:  {opt_gmean:.4f} (score: {score_gmean:.4f})")
    print(f"   YOUDEN OPTIMAL:  {opt_youden:.4f} (score: {score_youden:.4f})")
    
    # Use G-mean as primary
    opt_thresh = opt_gmean
    
    # Comparison table
    print("\n" + "="*80)
    print("  THRESHOLD COMPARISON")
    print("="*80)
    header = f"{'Thresh':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'NPV':>8} {'Spec':>8} {'FN':>8} {'FP':>8}"
    print(header)
    print("-"*80)
    
    thresholds = sorted(set([0.2, 0.3, 0.4, 0.5, opt_thresh, 0.6, 0.7, 0.8]))
    
    for thresh in thresholds:
        y_pred = (avg_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)  # type: ignore
        rec = recall_score(y_test, y_pred, zero_division=0)  # type: ignore
        f1 = f1_score(y_test, y_pred, zero_division=0)  # type: ignore
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        marker = " <-- OPTIMAL" if abs(thresh - opt_thresh) < 0.001 else ""
        print(f"{thresh:>8.4f} {acc:>7.2%} {prec:>7.2%} {rec:>7.2%} {f1:>7.2%} {npv:>7.2%} {spec:>7.2%} {fn:>8} {fp:>8}{marker}")
    
    print("="*80)
    
    # AUC-ROC
    auc = roc_auc_score(y_test, avg_proba)
    print(f"\nAUC-ROC: {auc:.4f}")
    
    # Save optimal threshold
    threshold_data = {
        'method': 'gmean',
        'threshold': float(opt_thresh),
        'score': float(score_gmean),
        'youden_threshold': float(opt_youden),
        'youden_score': float(score_youden),
        'auc_roc': float(auc),
        'test_samples': len(test_df),
        'test_class_balance': {'positive': int(n_pos), 'negative': int(n_neg)}
    }
    
    threshold_path = model_dir / "optimal_threshold.json"
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    
    print(f"\nSaved optimal threshold to {threshold_path}")
    print(f"\nUSE THIS THRESHOLD IN YOUR ML PREDICTOR: {opt_thresh:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()

