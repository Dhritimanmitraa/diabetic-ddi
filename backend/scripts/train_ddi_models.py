#!/usr/bin/env python
"""
Train DDI (Drug-Drug Interaction) prediction models.

Loads real drug and interaction data from the PostgreSQL / SQLite database,
runs Bayesian hyperparameter optimisation (Optuna TPE), trains an ensemble
of Random Forest + XGBoost + LightGBM, computes an optimal classification
threshold, and persists all artefacts to ``backend/models/``.

Usage (from the ``backend/`` directory):
    python -m scripts.train_ddi_models [--trials 50] [--no-compare]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure reproducible hashing
os.environ.setdefault("PYTHONHASHSEED", "0")

# ---------------------------------------------------------------------------
# Make ``app`` importable when running from ``backend/``
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml.trainer import DDITrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("train_ddi")


def _sync_database_url() -> str:
    """Derive a synchronous DB URL from the async one in settings."""
    url = get_settings().DATABASE_URL
    # aiosqlite -> pysqlite (stdlib)
    url = url.replace("+aiosqlite", "")
    # asyncpg  -> psycopg2 (or whatever sync driver is installed)
    url = url.replace("+asyncpg", "")
    return url


def main(n_trials: int = 50, run_comparison: bool = True) -> None:
    model_dir = os.path.join(_BACKEND_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # ---- 1. Connect to DB synchronously ----
    db_url = _sync_database_url()
    logger.info(f"Connecting to database: {db_url}")
    engine = create_engine(db_url, echo=False)

    with Session(engine) as session:
        trainer = DDITrainer(
            model_dir=model_dir,
            n_trials=n_trials,
            cv_folds=5,
            random_state=42,
        )

        # ---- 2. Load real data from DB ----
        drugs, interactions = trainer.load_data_from_db(session)

        if len(drugs) < 2:
            logger.error("Not enough drugs in DB to train.  Seed the database first.")
            sys.exit(1)
        if len(interactions) < 10:
            logger.error("Not enough interactions in DB (<10). Seed the database first.")
            sys.exit(1)

        # ---- 3. Prepare features & splits ----
        X_train, X_test, y_train, y_test = trainer.load_data_from_dicts(
            drugs, interactions
        )

        logger.info(
            f"Dataset ready — X_train: {X_train.shape}, X_test: {X_test.shape}"
        )

        # ---- 4. Train with Bayesian hyper-parameter tuning ----
        trainer.train_all_models(
            X_train, y_train,
            X_test, y_test,
            optimize=True,
            run_comparison=run_comparison,
        )

        # ---- 5. Save everything ----
        trainer.save_models()

        summary = trainer.get_training_summary()
        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"  Models trained : {summary['models_trained']}")
        logger.info(f"  Best model     : {summary.get('best_model')}")
        logger.info(f"  Best AUC-ROC   : {summary.get('best_auc', 0):.4f}")
        logger.info(f"  Artefacts saved: {model_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDI prediction models")
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Skip comparison of optimisation methods (faster)",
    )
    args = parser.parse_args()
    main(n_trials=args.trials, run_comparison=not args.no_compare)
