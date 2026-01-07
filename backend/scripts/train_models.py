"""
ML Model Training Script

Trains Drug-Drug Interaction prediction models with Bayesian hyperparameter optimization.

Usage:
    python -m scripts.train_models

    Or with options:
    python -m scripts.train_models --trials 100 --compare
"""

import asyncio
import sys
import os
import argparse
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import engine, async_session, init_db
from app.models import Drug, DrugInteraction
from app.ml.trainer import DDITrainer
from app.ml.models import ModelType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def load_data_from_db() -> tuple:
    """Load drugs and interactions from database."""
    async with async_session() as db:
        # Load drugs
        result = await db.execute(select(Drug))
        drugs_orm = result.scalars().all()

        drugs = [
            {
                "id": d.id,
                "name": d.name,
                "generic_name": d.generic_name,
                "drug_class": d.drug_class,
                "description": d.description,
                "mechanism": d.mechanism,
                "indication": d.indication,
                "molecular_weight": d.molecular_weight,
                "is_approved": d.is_approved,
            }
            for d in drugs_orm
        ]

        # Create ID to name mapping
        drug_id_to_name = {d.id: d.name for d in drugs_orm}

        # Load interactions
        result = await db.execute(select(DrugInteraction))
        interactions_orm = result.scalars().all()

        interactions = [
            {
                "drug1_name": drug_id_to_name.get(i.drug1_id, ""),
                "drug2_name": drug_id_to_name.get(i.drug2_id, ""),
                "severity": i.severity,
            }
            for i in interactions_orm
            if drug_id_to_name.get(i.drug1_id) and drug_id_to_name.get(i.drug2_id)
        ]

        return drugs, interactions


async def main(
    n_trials: int = 50,
    cv_folds: int = 5,
    run_comparison: bool = True,
    model_dir: str = "./models",
):
    """Main training function."""
    print()
    print("=" * 70)
    print("  DRUG-DRUG INTERACTION ML MODEL TRAINING")
    print("  with Bayesian Hyperparameter Optimization")
    print("=" * 70)
    print()
    print(f"  Configuration:")
    print(f"  • Optimization trials: {n_trials}")
    print(f"  • Cross-validation folds: {cv_folds}")
    print(f"  • Run comparison (Bayesian vs Grid vs Random): {run_comparison}")
    print(f"  • Model directory: {model_dir}")
    print()
    print("=" * 70)
    print()

    # Initialize database
    logger.info("Initializing database...")
    await init_db()

    # Load data
    logger.info("Loading data from database...")
    drugs, interactions = await load_data_from_db()

    if not drugs or not interactions:
        logger.error("No data found in database!")
        logger.error("Please run: python -m scripts.fetch_real_data first")
        return

    logger.info(f"Loaded {len(drugs)} drugs and {len(interactions)} interactions")

    # Check minimum data requirements
    if len(interactions) < 100:
        logger.warning(
            f"Only {len(interactions)} interactions found. Results may be unreliable."
        )

    # Create trainer
    trainer = DDITrainer(
        model_dir=model_dir,
        n_trials=n_trials,
        cv_folds=cv_folds,
        test_size=0.2,
        random_state=42,
        use_smote=True,
    )

    # Prepare data
    logger.info("Preparing training data...")
    X_train, X_test, y_train, y_test = trainer.load_data_from_dicts(drugs, interactions)

    logger.info(f"Training set: {len(y_train)} samples")
    logger.info(f"Test set: {len(y_test)} samples")

    # Train models
    logger.info("\nStarting model training...")

    trainer.train_all_models(
        X_train, y_train, X_test, y_test, optimize=True, run_comparison=run_comparison
    )

    # Save models
    logger.info("\nSaving models...")
    trainer.save_models()

    # Print summary
    summary = trainer.get_training_summary()

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"  Models Trained: {summary['models_trained']}")
    print(f"  Best Model: {summary['best_model']}")
    print(f"  Best AUC-ROC: {summary['best_auc']:.4f}")
    print()
    print("  Individual Model Performance:")
    print("  " + "-" * 50)

    for model_name, metrics in summary["model_metrics"].items():
        print(f"  {model_name}:")
        print(f"    • AUC-ROC:  {metrics.get('auc_roc', 0):.4f}")
        print(f"    • F1-Score: {metrics.get('f1_score', 0):.4f}")
        print(f"    • Accuracy: {metrics.get('accuracy', 0):.4f}")
        print()

    if run_comparison and summary.get("optimization_comparison"):
        print("  Optimization Method Comparison:")
        print("  " + "-" * 50)
        for model_type, comparison in summary["optimization_comparison"].items():
            winner = comparison.get("winner", "N/A")
            efficiency = comparison.get("efficiency_gain", {})
            print(f"  {model_type}: Winner = {winner}")
            if efficiency:
                print(
                    f"    • Trial reduction: {efficiency.get('trial_reduction_percent', 0):.1f}%"
                )
                print(
                    f"    • Time reduction: {efficiency.get('time_reduction_percent', 0):.1f}%"
                )
        print()

    print("  Models saved to:", model_dir)
    print()
    print("  To use predictions, call:")
    print("    POST /ml/predict with drug names")
    print()
    print("=" * 70)


def run():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train DDI prediction models with Bayesian optimization"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials per model (default: 50)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison between Bayesian, Grid, and Random search",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip optimization method comparison (faster)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help="Directory to save trained models (default: ./models)",
    )

    args = parser.parse_args()

    # Default is to run comparison unless --no-compare is specified
    run_comparison = not args.no_compare
    if args.compare:
        run_comparison = True

    asyncio.run(
        main(
            n_trials=args.trials,
            cv_folds=args.cv_folds,
            run_comparison=run_comparison,
            model_dir=args.model_dir,
        )
    )


if __name__ == "__main__":
    run()
