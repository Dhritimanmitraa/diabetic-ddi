"""
Optimized ML Training for TWOSIDES dataset with imbalanced-learn.

Optimizations:
- Chunked CSV loading for memory efficiency
- Subsampling option for faster iteration
- Reduced hyperparameter trials with smart defaults
- Early stopping for gradient boosting
- Parallel processing where possible
- Progress logging
- SMOTE/ADASYN oversampling for class imbalance
- Focus on NPV (Negative Predictive Value) for safety
"""
import os
import sys
import asyncio
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging
import time

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ML models on TWOSIDES data with imbalanced-learn')
    parser.add_argument('--max-samples', type=int, default=500000,
                        help='Max training samples (default: 500000)')
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Bayesian optimization trials (default: 10)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: fewer trials, simpler models')
    parser.add_argument('--skip-xgb', action='store_true',
                        help='Skip XGBoost training')
    parser.add_argument('--skip-lgb', action='store_true',
                        help='Skip LightGBM training')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores)')
    parser.add_argument('--resampling', type=str, default='smote',
                        choices=['none', 'smote', 'adasyn', 'borderline', 'svm', 'smote_tomek', 'smote_enn'],
                        help='Resampling method (default: smote)')
    parser.add_argument('--sampling-ratio', type=float, default=0.5,
                        help='Target ratio of minority class (0.5 = 50% balance, default: 0.5)')
    return parser.parse_args()


def load_training_data_optimized(data_dir: Path, max_samples: int = None):
    """Load training data with optional subsampling."""
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    test_path = data_dir / "test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    logger.info(f"Loading training data from {data_dir}")
    
    # Load train with optional sampling
    total_train = sum(1 for _ in open(train_path, encoding='utf-8')) - 1  # minus header
    
    if max_samples and total_train > max_samples:
        # Calculate skip probability
        skip_frac = 1 - (max_samples / total_train)
        logger.info(f"Subsampling train from {total_train} to ~{max_samples}")
        
        # Random sampling via skiprows
        np.random.seed(42)
        skip_rows = np.random.rand(total_train) < skip_frac
        skip_indices = [i+1 for i, skip in enumerate(skip_rows) if skip]  # +1 for header
        
        train_df = pd.read_csv(train_path, skiprows=skip_indices, encoding='utf-8')
    else:
        train_df = pd.read_csv(train_path, encoding='utf-8')
    
    # Load val and test fully (they're smaller)
    val_df = pd.read_csv(val_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    
    logger.info(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def prepare_features_fast(df: pd.DataFrame) -> tuple:
    """
    Fast feature preparation using vectorized operations.
    """
    n_samples = len(df)
    
    # Pre-allocate feature matrix
    # 2 (matched) + 50*2 (class) + 30*2 (mechanism) + 40*2 (name) = 242 features
    n_features = 2 + 100 + 60 + 80
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    
    # Boolean features (columns 0-1)
    X[:, 0] = df['drug1_matched'].fillna(0).astype(np.float32).values
    X[:, 1] = df['drug2_matched'].fillna(0).astype(np.float32).values
    
    # Vectorized hash encoding
    def text_hash_vectorized(series, n_features, offset):
        for idx, val in enumerate(series):
            if pd.notna(val) and val:
                s = str(val).lower()
                for j, char in enumerate(s[:20]):  # Limit string length
                    X[idx, offset + hash(char + str(j)) % n_features] += 1
    
    # Apply hash encoding for each text column
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


def train_random_forest(X_train, y_train, X_val, y_val, n_jobs=-1, fast=False):
    """Train Random Forest with good defaults."""
    logger.info("Training Random Forest...")
    start = time.time()
    
    if fast:
        params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'n_jobs': n_jobs,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    else:
        params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': n_jobs,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start
    logger.info(f"Random Forest trained in {elapsed:.1f}s")
    
    # Calibrate
    logger.info("Calibrating Random Forest...")
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_val, y_val)
    
    return calibrated, params


def train_xgboost(X_train, y_train, X_val, y_val, n_jobs=-1, fast=False):
    """Train XGBoost with early stopping."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed, skipping")
        return None, None
    
    logger.info("Training XGBoost...")
    start = time.time()
    
    # Calculate scale_pos_weight for imbalanced data
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    if fast:
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'n_jobs': n_jobs,
            'random_state': 42,
            'tree_method': 'hist',  # Fast histogram-based
            'early_stopping_rounds': 20,
            'eval_metric': 'auc'
        }
    else:
        params = {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'n_jobs': n_jobs,
            'random_state': 42,
            'tree_method': 'hist',
            'early_stopping_rounds': 30,
            'eval_metric': 'auc'
        }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    elapsed = time.time() - start
    logger.info(f"XGBoost trained in {elapsed:.1f}s (stopped at {model.best_iteration} trees)")
    
    # Calibrate
    logger.info("Calibrating XGBoost...")
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_val, y_val)
    
    return calibrated, params


def train_lightgbm(X_train, y_train, X_val, y_val, n_jobs=-1, fast=False):
    """Train LightGBM with early stopping."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not installed, skipping")
        return None, None
    
    logger.info("Training LightGBM...")
    start = time.time()
    
    # Calculate class weights
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    
    if fast:
        params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'n_jobs': n_jobs,
            'random_state': 42,
            'verbose': -1,
        }
        callbacks = [lgb.early_stopping(20, verbose=False)]
    else:
        params = {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.05,
            'num_leaves': 127,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'n_jobs': n_jobs,
            'random_state': 42,
            'verbose': -1,
        }
        callbacks = [lgb.early_stopping(30, verbose=False)]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    
    elapsed = time.time() - start
    logger.info(f"LightGBM trained in {elapsed:.1f}s (stopped at {model.best_iteration_} trees)")
    
    # Calibrate
    logger.info("Calibrating LightGBM...")
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated.fit(X_val, y_val)
    
    return calibrated, params


def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    """Evaluate a single model with focus on NPV."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate all metrics including NPV
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value (CRITICAL!)
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'threshold': threshold
    }
    
    logger.info(f"\n{model_name} Results:")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  PPV:       {metrics['ppv']:.4f}")
    logger.info(f"  NPV:       {metrics['npv']:.4f}")  # Highlight NPV
    logger.info(f"  Brier:     {metrics['brier_score']:.4f}")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return metrics


def ensemble_predict(models, X):
    """Simple averaging ensemble."""
    probas = []
    for model in models.values():
        if model is not None:
            probas.append(model.predict_proba(X)[:, 1])
    
    if not probas:
        raise ValueError("No models available for ensemble")
    
    avg_proba = np.mean(probas, axis=0)
    return (avg_proba >= 0.5).astype(int), avg_proba


async def save_metrics_to_db(results: dict):
    """Save training metrics to database."""
    from app.database import async_session, init_db
    from app.models import ModelMetrics
    
    await init_db()
    
    async with async_session() as db:
        for model_type, model_data in results.get('models', {}).items():
            metrics = model_data.get('metrics', {})
            
            metric_record = ModelMetrics(
                model_type=model_type,
                model_version=f"twosides_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                accuracy=metrics.get('accuracy'),
                precision=metrics.get('precision'),
                recall=metrics.get('recall'),
                f1_score=metrics.get('f1_score'),
                auc_roc=metrics.get('auc_roc'),
                training_samples=results.get('training_samples'),
                test_samples=results.get('test_samples'),
                n_features=results.get('n_features'),
                hyperparameters=json.dumps(model_data.get('params', {})),
                trained_at=datetime.now(timezone.utc)
            )
            db.add(metric_record)
        
        await db.commit()
        logger.info("Saved metrics to database")


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("OPTIMIZED ML TRAINING PIPELINE")
    print("=" * 60)
    print(f"Max samples: {args.max_samples}")
    print(f"Fast mode: {args.fast}")
    print(f"N jobs: {args.n_jobs}")
    print("=" * 60 + "\n")
    
    overall_start = time.time()
    
    data_dir = Path(__file__).parent.parent / "data" / "training"
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Load data
    train_df, val_df, test_df = load_training_data_optimized(
        data_dir, 
        max_samples=args.max_samples
    )
    
    # Prepare features
    logger.info("\nPreparing features (vectorized)...")
    feat_start = time.time()
    X_train, y_train = prepare_features_fast(train_df)
    X_val, y_val = prepare_features_fast(val_df)
    X_test, y_test = prepare_features_fast(test_df)
    logger.info(f"Features prepared in {time.time() - feat_start:.1f}s")
    
    logger.info(f"Feature shape: {X_train.shape}")
    logger.info(f"Train distribution BEFORE resampling: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
    logger.info(f"Class imbalance ratio: {np.sum(y_train==1) / np.sum(y_train==0):.4f}")
    
    # Apply resampling if requested
    if args.resampling != 'none':
        logger.info(f"\nApplying {args.resampling.upper()} resampling...")
        logger.info(f"Target sampling ratio: {args.sampling_ratio}")
        
        resampling_start = time.time()
        
        if args.resampling == 'smote':
            sampler = SMOTE(
                sampling_strategy=args.sampling_ratio,
                random_state=42,
                n_jobs=args.n_jobs,
                k_neighbors=5
            )
        elif args.resampling == 'adasyn':
            sampler = ADASYN(
                sampling_strategy=args.sampling_ratio,
                random_state=42,
                n_jobs=args.n_jobs,
                n_neighbors=5
            )
        elif args.resampling == 'borderline':
            sampler = BorderlineSMOTE(
                sampling_strategy=args.sampling_ratio,
                random_state=42,
                n_jobs=args.n_jobs,
                k_neighbors=5
            )
        elif args.resampling == 'svm':
            sampler = SVMSMOTE(
                sampling_strategy=args.sampling_ratio,
                random_state=42,
                n_jobs=args.n_jobs,
                k_neighbors=5
            )
        elif args.resampling == 'smote_tomek':
            sampler = SMOTETomek(
                sampling_strategy=args.sampling_ratio,
                random_state=42,
                n_jobs=args.n_jobs
            )
        elif args.resampling == 'smote_enn':
            sampler = SMOTEENN(
                sampling_strategy=args.sampling_ratio,
                random_state=42,
                n_jobs=args.n_jobs
            )
        
        try:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            logger.info(f"Resampling complete in {time.time() - resampling_start:.1f}s")
            logger.info(f"Train distribution AFTER resampling: 0={np.sum(y_train_resampled==0)}, 1={np.sum(y_train_resampled==1)}")
            logger.info(f"New class balance ratio: {np.sum(y_train_resampled==1) / np.sum(y_train_resampled==0):.4f}")
            
            # Update training data
            X_train = X_train_resampled
            y_train = y_train_resampled
            
            # Free memory
            del X_train_resampled, y_train_resampled, sampler
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            logger.warning("Continuing without resampling...")
    
    # Free memory
    del train_df, val_df, test_df
    
    models = {}
    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'training_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'fast_mode': args.fast,
        'resampling_method': args.resampling,
        'sampling_ratio': args.sampling_ratio,
        'train_class_distribution': {
            'class_0': int(np.sum(y_train == 0)),
            'class_1': int(np.sum(y_train == 1))
        },
        'models': {},
    }
    
    # Train Random Forest
    logger.info("\n" + "-" * 40)
    rf_model, rf_params = train_random_forest(
        X_train, y_train, X_val, y_val,
        n_jobs=args.n_jobs, fast=args.fast
    )
    models['random_forest'] = rf_model
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    results['models']['random_forest'] = {'params': rf_params, 'metrics': rf_metrics}
    
    # Train XGBoost
    if not args.skip_xgb:
        logger.info("\n" + "-" * 40)
        xgb_model, xgb_params = train_xgboost(
            X_train, y_train, X_val, y_val,
            n_jobs=args.n_jobs, fast=args.fast
        )
        if xgb_model:
            models['xgboost'] = xgb_model
            xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
            results['models']['xgboost'] = {'params': xgb_params, 'metrics': xgb_metrics}
    
    # Train LightGBM
    if not args.skip_lgb:
        logger.info("\n" + "-" * 40)
        lgb_model, lgb_params = train_lightgbm(
            X_train, y_train, X_val, y_val,
            n_jobs=args.n_jobs, fast=args.fast
        )
        if lgb_model:
            models['lightgbm'] = lgb_model
            lgb_metrics = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
            results['models']['lightgbm'] = {'params': lgb_params, 'metrics': lgb_metrics}
    
    # Ensemble
    if len(models) > 1:
        logger.info("\n" + "-" * 40)
        logger.info("Evaluating Ensemble...")
        y_pred_ens, y_proba_ens = ensemble_predict(models, X_test)
        
        # Confusion matrix for ensemble
        cm_ens = confusion_matrix(y_test, y_pred_ens)
        tn_ens, fp_ens, fn_ens, tp_ens = cm_ens.ravel()
        
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_ens),
            'precision': precision_score(y_test, y_pred_ens, zero_division=0),
            'recall': recall_score(y_test, y_pred_ens, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_ens, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba_ens),
            'brier_score': brier_score_loss(y_test, y_proba_ens),
            'specificity': tn_ens / (tn_ens + fp_ens) if (tn_ens + fp_ens) > 0 else 0,
            'sensitivity': tp_ens / (tp_ens + fn_ens) if (tp_ens + fn_ens) > 0 else 0,
            'ppv': tp_ens / (tp_ens + fp_ens) if (tp_ens + fp_ens) > 0 else 0,
            'npv': tn_ens / (tn_ens + fn_ens) if (tn_ens + fn_ens) > 0 else 0,  # CRITICAL!
            'confusion_matrix': {
                'tn': int(tn_ens),
                'fp': int(fp_ens),
                'fn': int(fn_ens),
                'tp': int(tp_ens)
            }
        }
        
        logger.info(f"\nEnsemble Results:")
        logger.info(f"  AUC-ROC:   {ensemble_metrics['auc_roc']:.4f}")
        logger.info(f"  F1-Score:  {ensemble_metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {ensemble_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {ensemble_metrics['recall']:.4f}")
        logger.info(f"  Specificity: {ensemble_metrics['specificity']:.4f}")
        logger.info(f"  Sensitivity: {ensemble_metrics['sensitivity']:.4f}")
        logger.info(f"  PPV:       {ensemble_metrics['ppv']:.4f}")
        logger.info(f"  NPV:       {ensemble_metrics['npv']:.4f} [CRITICAL METRIC]")
        logger.info(f"  Confusion Matrix: TN={tn_ens}, FP={fp_ens}, FN={fn_ens}, TP={tp_ens}")
        
        results['ensemble'] = ensemble_metrics
    
    # Save models
    logger.info("\n" + "-" * 40)
    logger.info("Saving models...")
    
    for model_name, model in models.items():
        if model is not None:
            model_path = model_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
    
    # Save results
    results_path = model_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save to DB
    logger.info("\nSaving metrics to database...")
    asyncio.run(save_metrics_to_db(results))
    
    total_time = time.time() - overall_start
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nModels saved to: {model_dir}")
    print("\nPerformance Summary:")
    print("-" * 60)
    print(f"{'Model':<15} {'AUC':>8} {'F1':>8} {'NPV':>8} {'Recall':>8} {'Precision':>10}")
    print("-" * 60)
    
    for model_name, model_data in results['models'].items():
        m = model_data['metrics']
        print(f"{model_name:<15} {m['auc_roc']:>8.4f} {m['f1_score']:>8.4f} {m.get('npv', 0):>8.4f} {m.get('recall', 0):>8.4f} {m.get('precision', 0):>10.4f}")
    
    if 'ensemble' in results:
        e = results['ensemble']
        print("-" * 60)
        print(f"{'ENSEMBLE':<15} {e['auc_roc']:>8.4f} {e['f1_score']:>8.4f} {e.get('npv', 0):>8.4f} {e.get('recall', 0):>8.4f} {e.get('precision', 0):>10.4f}")
        print("-" * 60)
        print(f"\nWARNING: NPV (Negative Predictive Value) is CRITICAL for safety!")
        print(f"   Current NPV: {e.get('npv', 0):.2%}")
        if e.get('npv', 0) < 0.5:
            print(f"   WARNING: NPV < 50% - Model is unsafe for 'safe' predictions!")
        elif e.get('npv', 0) < 0.7:
            print(f"   CAUTION: NPV < 70% - Consider improving further")
        else:
            print(f"   NPV is acceptable for safety-critical applications")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
