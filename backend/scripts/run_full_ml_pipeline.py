#!/usr/bin/env python
"""
Run the complete DDI ML lifecycle in one command.

Stages:
1) Data cleaning
2) Exploratory data analysis (EDA)
3) Feature engineering
4) Model training
5) Evaluation
6) Deployment artifact generation

Usage:
    python -m scripts.run_full_ml_pipeline --trials 30
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# Make app importable when run from backend/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.config import get_settings
from app.models import Drug, DrugInteraction
from app.ml.feature_engineering import DrugFeatureExtractor, prepare_training_data
from app.ml.trainer import DDITrainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger("full_ml_pipeline")


VALID_SEVERITIES = {"minor", "moderate", "major", "contraindicated"}


def _sync_database_url() -> str:
    """Derive a synchronous DB URL from the async one in settings."""
    url = get_settings().DATABASE_URL
    url = url.replace("+aiosqlite", "")
    url = url.replace("+asyncpg", "")
    return url


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_dataframes(db_url: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load drugs and interactions from DB into dataframes."""
    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        drugs_orm = session.execute(select(Drug)).scalars().all()
        interactions_orm = session.execute(select(DrugInteraction)).scalars().all()

    drugs = [
        {
            "id": d.id,
            "name": (d.name or "").strip(),
            "generic_name": (d.generic_name or "").strip(),
            "drug_class": (d.drug_class or "Unknown").strip() or "Unknown",
            "description": (d.description or "").strip(),
            "mechanism": (d.mechanism or "").strip(),
            "indication": (d.indication or "").strip(),
            "molecular_weight": _to_float(d.molecular_weight, 0.0),
            "is_approved": bool(d.is_approved),
        }
        for d in drugs_orm
    ]

    id_to_name = {d["id"]: d["name"] for d in drugs}
    interactions = [
        {
            "id": i.id,
            "drug1_id": i.drug1_id,
            "drug2_id": i.drug2_id,
            "drug1_name": id_to_name.get(i.drug1_id, "").strip(),
            "drug2_name": id_to_name.get(i.drug2_id, "").strip(),
            "severity": (i.severity or "moderate").strip().lower(),
            "description": (i.description or "").strip(),
            "confidence_score": _to_float(i.confidence_score, 0.8),
        }
        for i in interactions_orm
    ]

    return pd.DataFrame(drugs), pd.DataFrame(interactions)


def clean_data(
    drugs_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Clean core training tables and return cleaning report."""
    report: Dict[str, Any] = {
        "before": {
            "drugs": int(len(drugs_df)),
            "interactions": int(len(interactions_df)),
        }
    }

    # Drug cleaning
    drugs_df = drugs_df.copy()
    drugs_df["name_norm"] = drugs_df["name"].str.strip().str.lower()
    drugs_df = drugs_df[drugs_df["name_norm"] != ""]
    drugs_df = drugs_df.drop_duplicates(subset=["name_norm"], keep="first")

    text_cols = ["generic_name", "drug_class", "description", "mechanism", "indication"]
    for col in text_cols:
        drugs_df[col] = drugs_df[col].fillna("").astype(str).str.strip()
    drugs_df["drug_class"] = drugs_df["drug_class"].replace("", "Unknown")
    drugs_df["molecular_weight"] = pd.to_numeric(
        drugs_df["molecular_weight"], errors="coerce"
    ).fillna(0.0)

    # Interaction cleaning
    interactions_df = interactions_df.copy()
    interactions_df["drug1_name_norm"] = interactions_df["drug1_name"].str.strip().str.lower()
    interactions_df["drug2_name_norm"] = interactions_df["drug2_name"].str.strip().str.lower()

    interactions_df = interactions_df[
        (interactions_df["drug1_name_norm"] != "")
        & (interactions_df["drug2_name_norm"] != "")
        & (interactions_df["drug1_name_norm"] != interactions_df["drug2_name_norm"])
    ]

    interactions_df["severity"] = interactions_df["severity"].where(
        interactions_df["severity"].isin(VALID_SEVERITIES), "moderate"
    )

    # Canonical pair ordering removes mirrored duplicates
    pair_min = interactions_df[["drug1_name_norm", "drug2_name_norm"]].min(axis=1)
    pair_max = interactions_df[["drug1_name_norm", "drug2_name_norm"]].max(axis=1)
    interactions_df["pair_key"] = pair_min + "__" + pair_max
    interactions_df = interactions_df.drop_duplicates(subset=["pair_key"], keep="first")

    report["after"] = {
        "drugs": int(len(drugs_df)),
        "interactions": int(len(interactions_df)),
    }
    report["dropped"] = {
        "drugs": report["before"]["drugs"] - report["after"]["drugs"],
        "interactions": report["before"]["interactions"] - report["after"]["interactions"],
    }

    keep_drug_cols = [
        "id",
        "name",
        "generic_name",
        "drug_class",
        "description",
        "mechanism",
        "indication",
        "molecular_weight",
        "is_approved",
    ]
    keep_interaction_cols = [
        "id",
        "drug1_id",
        "drug2_id",
        "drug1_name",
        "drug2_name",
        "severity",
        "description",
        "confidence_score",
    ]
    return (
        drugs_df[keep_drug_cols].reset_index(drop=True),
        interactions_df[keep_interaction_cols].reset_index(drop=True),
        report,
    )


def run_eda(drugs_df: pd.DataFrame, interactions_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute compact EDA summary stats."""
    missing_pct = (drugs_df.isna().mean() * 100).round(2).to_dict()
    class_counts = drugs_df["drug_class"].value_counts().head(15).to_dict()
    severity_counts = interactions_df["severity"].value_counts().to_dict()

    degree_counter: Dict[str, int] = {}
    for row in interactions_df.itertuples(index=False):
        d1 = str(row.drug1_name)
        d2 = str(row.drug2_name)
        degree_counter[d1] = degree_counter.get(d1, 0) + 1
        degree_counter[d2] = degree_counter.get(d2, 0) + 1
    top_degree = sorted(degree_counter.items(), key=lambda item: item[1], reverse=True)[:20]

    return {
        "dataset": {
            "n_drugs": int(len(drugs_df)),
            "n_interactions": int(len(interactions_df)),
            "unique_drug_classes": int(drugs_df["drug_class"].nunique()),
        },
        "missing_percent_by_column": missing_pct,
        "top_drug_classes": class_counts,
        "severity_distribution": severity_counts,
        "top_interaction_degree_drugs": [
            {"drug": drug, "degree": degree} for drug, degree in top_degree
        ],
    }


def run_feature_engineering(
    drugs_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    random_seed: int,
) -> Tuple[DrugFeatureExtractor, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit feature extractor and create training matrix."""
    drugs_records = drugs_df.to_dict(orient="records")
    interactions_records = interactions_df.to_dict(orient="records")

    extractor = DrugFeatureExtractor()
    extractor.fit(drugs_records, interactions_records)

    # Keep deterministic negative sampling
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)

    drug_pairs, labels = prepare_training_data(
        drugs_records,
        interactions_records,
        negative_ratio=1.0,
    )
    X = extractor.extract_batch_features(drug_pairs)
    y = np.array(labels)

    class_values, class_counts = np.unique(y, return_counts=True)
    class_dist = {str(int(cls)): int(cnt) for cls, cnt in zip(class_values, class_counts)}
    fe_report = {
        "n_pairs": int(len(drug_pairs)),
        "n_features": int(X.shape[1]) if X.size else 0,
        "class_distribution": class_dist,
        "feature_names": extractor.get_feature_names(),
    }
    return extractor, X, y, fe_report


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_pipeline(trials: int, no_smote: bool, run_comparison: bool) -> Dict[str, Any]:
    """Run all ML lifecycle stages and persist artifacts."""
    started_at = datetime.now(timezone.utc)
    random_seed = 42

    reports_dir = os.path.join(_BACKEND_DIR, "models", "reports")
    models_dir = os.path.join(_BACKEND_DIR, "models")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    logger.info("[1/6] Loading raw data")
    db_url = _sync_database_url()
    drugs_df, interactions_df = load_dataframes(db_url)
    if len(drugs_df) < 2 or len(interactions_df) < 10:
        raise RuntimeError(
            "Insufficient data in DB. Need at least 2 drugs and 10 interactions."
        )

    logger.info("[2/6] Data cleaning")
    clean_drugs_df, clean_interactions_df, cleaning_report = clean_data(drugs_df, interactions_df)
    _write_json(os.path.join(reports_dir, "cleaning_report.json"), cleaning_report)

    logger.info("[3/6] EDA")
    eda_report = run_eda(clean_drugs_df, clean_interactions_df)
    _write_json(os.path.join(reports_dir, "eda_report.json"), eda_report)

    logger.info("[4/6] Feature engineering")
    _, X, y, fe_report = run_feature_engineering(
        clean_drugs_df, clean_interactions_df, random_seed=random_seed
    )
    _write_json(os.path.join(reports_dir, "feature_engineering_report.json"), fe_report)

    logger.info("[5/6] Model training + evaluation")
    trainer = DDITrainer(
        model_dir=models_dir,
        n_trials=trials,
        cv_folds=5,
        test_size=0.2,
        random_state=random_seed,
        use_smote=not no_smote,
    )
    # Re-use cleaned records for robust training persistence in existing format.
    X_train, X_test, y_train, y_test = trainer.load_data_from_dicts(
        clean_drugs_df.to_dict(orient="records"),
        clean_interactions_df.to_dict(orient="records"),
    )
    trainer.train_all_models(
        X_train,
        y_train,
        X_test,
        y_test,
        optimize=True,
        run_comparison=run_comparison,
    )
    trainer.save_models()
    training_summary = trainer.get_training_summary()

    # Save extra matrix-level report from stage 4 for quick audit
    eval_report = {
        "matrix_shape": {"X_rows": int(X.shape[0]), "X_cols": int(X.shape[1]) if X.size else 0},
        "labels": {
            "positive": int(np.sum(y == 1)),
            "negative": int(np.sum(y == 0)),
        },
        "trainer_summary": training_summary,
    }
    _write_json(os.path.join(reports_dir, "evaluation_report.json"), eval_report)

    logger.info("[6/6] Deployment artifact generation")
    deployment_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_dir": models_dir,
        "required_files": [
            "feature_extractor.pkl",
            "training_results.json",
            "optimal_threshold.json",
            "random_forest_model.pkl",
            "xgboost_model.pkl",
            "lightgbm_model.pkl",
        ],
        "api_endpoints": {
            "status": "/ml/status",
            "predict": "/ml/predict",
            "trigger_train": "/ml/train",
        },
        "best_model": training_summary.get("best_model"),
        "best_auc": training_summary.get("best_auc", 0),
    }
    _write_json(os.path.join(models_dir, "deployment_manifest.json"), deployment_manifest)

    finished_at = datetime.now(timezone.utc)
    final_summary = {
        "status": "completed",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        "reports_dir": reports_dir,
        "models_dir": models_dir,
        "training_summary": training_summary,
    }
    _write_json(os.path.join(reports_dir, "pipeline_summary.json"), final_summary)
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full DDI ML lifecycle")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model")
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE balancing during model training",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip Bayesian/Grid/Random comparison to run faster",
    )
    args = parser.parse_args()

    summary = run_pipeline(
        trials=args.trials,
        no_smote=args.no_smote,
        run_comparison=not args.no_compare,
    )
    logger.info("Pipeline completed: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
