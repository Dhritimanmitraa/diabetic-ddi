"""
Comprehensive Model Evaluation Script.

Evaluates trained ML models on the test set and provides detailed metrics.
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import time
import joblib

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_features_fast(df: pd.DataFrame) -> tuple:
    """
    Fast feature preparation using vectorized operations.
    Must match training script exactly.
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


def load_models(model_dir: Path):
    """Load all trained models."""
    models = {}
    model_names = ['random_forest', 'xgboost', 'lightgbm']
    
    for model_name in model_names:
        model_path = model_dir / f"{model_name}_model.pkl"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        try:
            model = joblib.load(model_path)
            
            # Handle both DDIModel format and direct sklearn models
            if isinstance(model, dict) and 'model' in model:
                models[model_name] = model.get('calibrated_model') or model.get('model')
            else:
                models[model_name] = model
            
            logger.info(f"✓ Loaded {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
    
    return models


def evaluate_model(model, X_test, y_test, model_name: str, threshold: float = 0.5):
    """Evaluate a single model with comprehensive metrics."""
    logger.info(f"\nEvaluating {model_name}...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Binary predictions with threshold
    y_pred_binary = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary, zero_division=0),
        'recall': recall_score(y_test, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_test, y_pred_binary, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['confusion_matrix'] = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    metrics['ppv'] = metrics['precision']  # Positive predictive value
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value
    
    return metrics, y_proba


def ensemble_predict(models, X, threshold: float = 0.5):
    """Simple averaging ensemble."""
    probas = []
    
    for model_name, model in models.items():
        try:
            proba = model.predict_proba(X)[:, 1]
            probas.append(proba)
        except Exception as e:
            logger.warning(f"Ensemble: {model_name} failed: {e}")
    
    if not probas:
        raise ValueError("No models available for ensemble")
    
    avg_proba = np.mean(probas, axis=0)
    pred = (avg_proba >= threshold).astype(int)
    
    return pred, avg_proba


def find_optimal_threshold(y_true, y_proba, method='gmean'):
    """
    Find optimal classification threshold using various methods.
    
    Methods:
    - 'gmean': Geometric mean (sqrt(TPR * (1-FPR))) - balanced
    - 'youden': Youden's J statistic (TPR - FPR) - maximize sensitivity + specificity
    - 'f1': Maximize F1-score
    - 'balanced': Balance precision and recall for safety
    
    Returns: (optimal_threshold, metrics_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    if method == 'gmean':
        # G-Mean: sqrt(TPR * (1 - FPR))
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        best_thresh = thresholds[ix]
        score = gmeans[ix]
        
    elif method == 'youden':
        # Youden's J: TPR - FPR (same as sensitivity + specificity - 1)
        j_scores = tpr - fpr
        ix = np.argmax(j_scores)
        best_thresh = thresholds[ix]
        score = j_scores[ix]
        
    elif method == 'f1':
        # Find threshold that maximizes F1
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        ix = np.argmax(f1_scores[:-1])  # Last value is undefined
        best_thresh = pr_thresholds[ix]
        score = f1_scores[ix]
        
    elif method == 'balanced':
        # For medical safety: prioritize catching negatives (high NPV)
        # Lower threshold = more negative predictions = fewer false negatives
        # Find threshold where specificity >= 0.5 while maximizing sensitivity
        specificity = 1 - fpr
        # Want high sensitivity (TPR) with decent specificity
        balanced_scores = 0.7 * tpr + 0.3 * specificity  # Weight toward sensitivity
        ix = np.argmax(balanced_scores)
        best_thresh = thresholds[ix]
        score = balanced_scores[ix]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return best_thresh, score


def evaluate_at_multiple_thresholds(y_true, y_proba, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Evaluate metrics at multiple thresholds to show the trade-off."""
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'threshold': thresh,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'fn': fn,
            'fp': fp
        })
    
    return results


def print_metrics_table(metrics_dict: dict, title: str = "Model Performance"):
    """Print a formatted metrics table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Value':>15} {'Unit':>10}")
    print(f"{'-'*80}")
    
    metric_labels = {
        'accuracy': ('Accuracy', '%'),
        'precision': ('Precision', '%'),
        'recall': ('Recall (Sensitivity)', '%'),
        'f1_score': ('F1-Score', '%'),
        'auc_roc': ('AUC-ROC', ''),
        'brier_score': ('Brier Score', ''),
        'specificity': ('Specificity', '%'),
        'ppv': ('PPV', '%'),
        'npv': ('NPV', '%'),
    }
    
    for key, (label, unit) in metric_labels.items():
        if key in metrics_dict:
            val = metrics_dict[key]
            if unit == '%':
                print(f"{label:<20} {val*100:>14.2f}%")
            else:
                print(f"{label:<20} {val:>15.6f}")
    
    # Confusion matrix
    if 'confusion_matrix' in metrics_dict:
        cm = metrics_dict['confusion_matrix']
        print(f"\n{'Confusion Matrix':<20}")
        print(f"  True Negatives:  {cm['true_negatives']:>10,}")
        print(f"  False Positives: {cm['false_positives']:>10,}")
        print(f"  False Negatives: {cm['false_negatives']:>10,}")
        print(f"  True Positives:  {cm['true_positives']:>10,}")
    
    print(f"{'='*80}\n")


def compare_with_saved(saved_results: dict, current_metrics: dict, model_name: str):
    """Compare current metrics with saved training metrics."""
    if 'models' not in saved_results:
        return
    
    if model_name not in saved_results['models']:
        return
    
    saved_metrics = saved_results['models'][model_name].get('metrics', {})
    
    print(f"\n{'─'*80}")
    print(f"  Comparison: {model_name.upper()} (Saved vs Current)")
    print(f"{'─'*80}")
    print(f"{'Metric':<20} {'Saved':>15} {'Current':>15} {'Diff':>15}")
    print(f"{'─'*80}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'brier_score']:
        if metric in saved_metrics and metric in current_metrics:
            saved_val = saved_metrics[metric]
            curr_val = current_metrics[metric]
            diff = curr_val - saved_val
            
            if metric == 'brier_score':
                # Lower is better for Brier score
                print(f"{metric:<20} {saved_val:>15.6f} {curr_val:>15.6f} {diff:>15.6f}")
            else:
                print(f"{metric:<20} {saved_val*100:>14.2f}% {curr_val*100:>14.2f}% {diff*100:>14.2f}%")
    
    print(f"{'─'*80}\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained ML models')
    parser.add_argument('--test-samples', type=int, default=None,
                        help='Limit test samples (for quick evaluation)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with saved training metrics')
    parser.add_argument('--find-optimal', action='store_true',
                        help='Find and save optimal threshold')
    parser.add_argument('--threshold-method', type=str, default='gmean',
                        choices=['gmean', 'youden', 'f1', 'balanced'],
                        help='Method for finding optimal threshold')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  MODEL EVALUATION REPORT")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Threshold: {args.threshold}")
    print("="*80)
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data" / "training"
    model_dir = Path(__file__).parent.parent / "models"
    
    # Load test data
    test_path = data_dir / "test.csv"
    if not test_path.exists():
        logger.error(f"Test data not found at {test_path}")
        return
    
    logger.info(f"Loading test data from {test_path}...")
    start_time = time.time()
    
    if args.test_samples:
        # Sample for quick evaluation
        total_lines = sum(1 for _ in open(test_path, encoding='utf-8')) - 1
        if total_lines > args.test_samples:
            skip_frac = 1 - (args.test_samples / total_lines)
            np.random.seed(42)
            skip_rows = np.random.rand(total_lines) < skip_frac
            skip_indices = [i+1 for i, skip in enumerate(skip_rows) if skip]
            test_df = pd.read_csv(test_path, skiprows=skip_indices, encoding='utf-8')
            logger.info(f"Sampled {len(test_df)} rows from {total_lines} total")
        else:
            test_df = pd.read_csv(test_path, encoding='utf-8')
    else:
        test_df = pd.read_csv(test_path, encoding='utf-8')
    
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Prepare features
    logger.info("Preparing features...")
    X_test, y_test = prepare_features_fast(test_df)
    logger.info(f"Feature shape: {X_test.shape}")
    logger.info(f"Class distribution: 0={np.sum(y_test==0):,}, 1={np.sum(y_test==1):,}")
    
    # Load models
    logger.info("\nLoading models...")
    models = load_models(model_dir)
    
    if not models:
        logger.error("No models loaded!")
        return
    
    # Load saved results for comparison
    saved_results = {}
    results_path = model_dir / "training_results.json"
    if results_path.exists() and args.compare:
        with open(results_path, 'r') as f:
            saved_results = json.load(f)
    
    # Evaluate each model
    all_metrics = {}
    all_probas = {}
    
    for model_name, model in models.items():
        metrics, probas = evaluate_model(model, X_test, y_test, model_name, args.threshold)
        all_metrics[model_name] = metrics
        all_probas[model_name] = probas
        
        print_metrics_table(metrics, f"{model_name.upper()} Model")
        
        if args.compare:
            compare_with_saved(saved_results, metrics, model_name)
    
    # Evaluate ensemble
    if len(models) > 1:
        logger.info("\nEvaluating Ensemble...")
        y_pred_ens, y_proba_ens = ensemble_predict(models, X_test, args.threshold)
        
        ensemble_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_ens),
            'precision': precision_score(y_test, y_pred_ens, zero_division=0),
            'recall': recall_score(y_test, y_pred_ens, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_ens, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba_ens),
            'brier_score': brier_score_loss(y_test, y_proba_ens),
        }
        
        cm_ens = confusion_matrix(y_test, y_pred_ens)
        tn, fp, fn, tp = cm_ens.ravel()
        ensemble_metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        ensemble_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ensemble_metrics['sensitivity'] = ensemble_metrics['recall']
        ensemble_metrics['ppv'] = ensemble_metrics['precision']
        ensemble_metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        print_metrics_table(ensemble_metrics, "ENSEMBLE (Average)")
        
        if args.compare and 'ensemble' in saved_results:
            compare_with_saved(saved_results, ensemble_metrics, 'ensemble')
    
    # Summary comparison table
    print("\n" + "="*80)
    print("  SUMMARY COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':>12} {'F1-Score':>12} {'AUC-ROC':>12} {'Precision':>12} {'Recall':>12}")
    print("─"*80)
    
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        if model_name in all_metrics:
            m = all_metrics[model_name]
            print(f"{model_name:<20} {m['accuracy']*100:>11.2f}% {m['f1_score']*100:>11.2f}% "
                  f"{m['auc_roc']:>11.4f} {m['precision']*100:>11.2f}% {m['recall']*100:>11.2f}%")
    
    if len(models) > 1:
        e = ensemble_metrics
        print(f"{'ENSEMBLE':<20} {e['accuracy']*100:>11.2f}% {e['f1_score']*100:>11.2f}% "
              f"{e['auc_roc']:>11.4f} {e['precision']*100:>11.2f}% {e['recall']*100:>11.2f}%")
    
    print("="*80)
    
    # ==================== OPTIMAL THRESHOLD ANALYSIS ====================
    optimal_thresholds = {}
    
    if args.find_optimal:
        print("\n" + "="*80)
        print("  OPTIMAL THRESHOLD ANALYSIS")
        print("="*80)
        
        # Use ensemble probabilities if available, else best single model
        if len(models) > 1:
            analysis_proba = y_proba_ens
            analysis_name = "Ensemble"
        else:
            # Use model with best AUC
            best_model = max(all_metrics.keys(), key=lambda k: all_metrics[k]['auc_roc'])
            analysis_proba = all_probas[best_model]
            analysis_name = best_model
        
        print(f"\nAnalyzing thresholds for: {analysis_name}")
        print(f"Method: {args.threshold_method}")
        
        # Find optimal threshold
        opt_thresh, opt_score = find_optimal_threshold(
            y_test, analysis_proba, method=args.threshold_method
        )
        
        print(f"\n✓ Optimal Threshold: {opt_thresh:.4f} (score: {opt_score:.4f})")
        
        # Show comparison at different thresholds
        print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
              f"{'F1':>10} {'NPV':>10} {'FN':>10} {'FP':>10}")
        print("─"*90)
        
        test_thresholds = [0.2, 0.3, 0.4, opt_thresh, 0.5, 0.6, 0.7, 0.8]
        test_thresholds = sorted(set(test_thresholds))
        
        multi_results = evaluate_at_multiple_thresholds(y_test, analysis_proba, test_thresholds)
        
        for r in multi_results:
            marker = " ←OPTIMAL" if abs(r['threshold'] - opt_thresh) < 0.001 else ""
            print(f"{r['threshold']:>10.4f} {r['accuracy']*100:>9.2f}% {r['precision']*100:>9.2f}% "
                  f"{r['recall']*100:>9.2f}% {r['f1']*100:>9.2f}% {r['npv']*100:>9.2f}% "
                  f"{r['fn']:>10,} {r['fp']:>10,}{marker}")
        
        print("="*80)
        
        # Evaluate at optimal threshold
        print(f"\n  METRICS AT OPTIMAL THRESHOLD ({opt_thresh:.4f})")
        print("="*80)
        
        y_pred_opt = (analysis_proba >= opt_thresh).astype(int)
        cm_opt = confusion_matrix(y_test, y_pred_opt)
        tn, fp, fn, tp = cm_opt.ravel()
        
        opt_metrics = {
            'threshold': opt_thresh,
            'accuracy': accuracy_score(y_test, y_pred_opt),
            'precision': precision_score(y_test, y_pred_opt, zero_division=0),
            'recall': recall_score(y_test, y_pred_opt, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_opt, zero_division=0),
            'auc_roc': roc_auc_score(y_test, analysis_proba),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        print_metrics_table(opt_metrics, f"Optimal ({args.threshold_method.upper()})")
        
        # Compare improvement
        print("\n  IMPROVEMENT vs DEFAULT THRESHOLD (0.5)")
        print("─"*80)
        
        default_metrics = all_metrics.get('random_forest', list(all_metrics.values())[0])
        
        print(f"{'Metric':<20} {'Default (0.5)':>15} {'Optimal':>15} {'Change':>15}")
        print("─"*80)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'npv', 'specificity']:
            if metric in default_metrics and metric in opt_metrics:
                default_val = default_metrics[metric]
                opt_val = opt_metrics[metric]
                change = opt_val - default_val
                arrow = "↑" if change > 0 else "↓" if change < 0 else "="
                print(f"{metric:<20} {default_val*100:>14.2f}% {opt_val*100:>14.2f}% "
                      f"{arrow} {abs(change)*100:>12.2f}%")
        
        # False negatives comparison
        default_fn = default_metrics.get('confusion_matrix', {}).get('false_negatives', 0)
        opt_fn = opt_metrics['confusion_matrix']['false_negatives']
        fn_reduction = default_fn - opt_fn
        print(f"{'False Negatives':<20} {default_fn:>15,} {opt_fn:>15,} "
              f"{'↓' if fn_reduction > 0 else '↑'} {abs(fn_reduction):>12,}")
        
        print("="*80)
        
        # Store optimal threshold for saving
        optimal_thresholds = {
            'method': args.threshold_method,
            'threshold': float(opt_thresh),
            'score': float(opt_score),
            'metrics_at_optimal': opt_metrics
        }
    
    # Save evaluation results
    eval_results = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': len(X_test),
        'threshold': args.threshold,
        'models': all_metrics,
        'ensemble': ensemble_metrics if len(models) > 1 else None,
        'optimal_threshold': optimal_thresholds if optimal_thresholds else None
    }
    
    eval_path = model_dir / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    
    # Save optimal threshold separately for easy loading by predictor
    if optimal_thresholds:
        threshold_path = model_dir / "optimal_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        logger.info(f"✓ Optimal threshold saved to {threshold_path}")
    
    logger.info(f"\n✓ Evaluation complete in {time.time() - start_time:.1f}s")
    logger.info(f"✓ Results saved to {eval_path}")
    
    print("\n" + "="*80)
    print("  ✅ EVALUATION COMPLETE")
    print("="*80)
    
    if optimal_thresholds:
        print(f"\n  🎯 USE THIS THRESHOLD IN YOUR ML PREDICTOR: {optimal_thresholds['threshold']:.4f}")
        print(f"     Method: {args.threshold_method}")
        print("="*80)


if __name__ == "__main__":
    main()

