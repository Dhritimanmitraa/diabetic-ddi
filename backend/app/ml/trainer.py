"""
Training Pipeline for DDI Prediction Models.

Handles data loading, feature extraction, model training,
and saving of trained models and metrics.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging

from app.ml.feature_engineering import DrugFeatureExtractor, prepare_training_data
from app.ml.models import ModelType, DDIModelFactory, DDIModel, EnsemblePredictor
from app.ml.bayesian_optimizer import (
    BayesianOptimizer,
    OptimizationComparator,
    OptimizationMethod,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


class DDITrainer:
    """
    Training pipeline for Drug-Drug Interaction models.

    Handles the complete training workflow:
    1. Load data from database
    2. Extract features
    3. Optimize hyperparameters
    4. Train final models
    5. Evaluate and save
    """

    def __init__(
        self,
        model_dir: str = "./models",
        n_trials: int = 100,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        use_smote: bool = True,
    ):
        self.model_dir = model_dir
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.use_smote = use_smote

        self.feature_extractor = DrugFeatureExtractor()
        self.models: Dict[ModelType, DDIModel] = {}
        self.optimization_results: Dict[ModelType, OptimizationResult] = {}
        self.comparison_results: Dict[ModelType, Dict] = {}
        self.training_metrics = {}

        os.makedirs(model_dir, exist_ok=True)

    def load_data_from_db(self, db_session) -> Tuple[List[Dict], List[Dict]]:
        """
        Load drugs and interactions from database.

        Args:
            db_session: SQLAlchemy async session

        Returns:
            Tuple of (drugs_list, interactions_list)
        """
        # This will be called with actual database session
        # For now, return placeholder
        raise NotImplementedError("Use load_data_from_dicts instead")

    def load_data_from_dicts(
        self, drugs: List[Dict], interactions: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data from dictionaries.

        Args:
            drugs: List of drug dictionaries
            interactions: List of interaction dictionaries

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(
            f"Loading data: {len(drugs)} drugs, {len(interactions)} interactions"
        )

        # Fit feature extractor
        self.feature_extractor.fit(drugs, interactions)

        # Prepare training data with positive and negative samples
        drug_pairs, labels = prepare_training_data(
            drugs, interactions, negative_ratio=1.0
        )

        # Extract features
        logger.info("Extracting features...")
        X = self.feature_extractor.extract_batch_features(drug_pairs)
        y = np.array(labels)

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Apply SMOTE for handling class imbalance
        if self.use_smote:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Class distribution: {np.bincount(y_train)}")

        return X_train, X_test, y_train, y_test

    def train_single_model(
        self,
        model_type: ModelType,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        optimize: bool = True,
        run_comparison: bool = False,
    ) -> DDIModel:
        """
        Train a single model type.

        Args:
            model_type: Type of model to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            optimize: Whether to run hyperparameter optimization
            run_comparison: Whether to compare optimization methods

        Returns:
            Trained model
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.value}")
        logger.info(f"{'='*60}")

        best_params = {}

        if optimize:
            if run_comparison:
                # Run comparison of optimization methods
                comparator = OptimizationComparator(
                    model_type,
                    n_trials_bayesian=self.n_trials,
                    n_trials_random=self.n_trials * 2,
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                )

                results = comparator.run_comparison(X_train, y_train)
                self.comparison_results[model_type] = (
                    comparator.get_comparison_summary()
                )

                # Use Bayesian result
                self.optimization_results[model_type] = results[
                    OptimizationMethod.BAYESIAN
                ]
                best_params = results[OptimizationMethod.BAYESIAN].best_params

                # Save comparison results
                comparator.save_results(
                    os.path.join(self.model_dir, f"{model_type.value}_comparison.json")
                )
            else:
                # Just run Bayesian optimization
                optimizer = BayesianOptimizer(
                    model_type,
                    n_trials=self.n_trials,
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                )

                result = optimizer.optimize(X_train, y_train)
                self.optimization_results[model_type] = result
                best_params = result.best_params

        # Train final model with best params
        best_params["random_state"] = self.random_state
        model = DDIModelFactory.create(model_type, best_params)
        model.train(X_train, y_train)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"\nTest Set Metrics for {model_type.value}:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Store
        self.models[model_type] = model

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        optimize: bool = True,
        run_comparison: bool = True,
    ) -> Dict[ModelType, DDIModel]:
        """
        Train all model types.

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            optimize: Whether to run hyperparameter optimization
            run_comparison: Whether to compare optimization methods

        Returns:
            Dictionary of trained models
        """
        model_types = [
            ModelType.RANDOM_FOREST,
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
        ]

        for model_type in model_types:
            self.train_single_model(
                model_type,
                X_train,
                y_train,
                X_test,
                y_test,
                optimize=optimize,
                run_comparison=run_comparison,
            )

        # Calculate ensemble performance
        if len(self.models) > 1:
            logger.info("\n" + "=" * 60)
            logger.info("Evaluating Ensemble Model")
            logger.info("=" * 60)

            ensemble = EnsemblePredictor(self.models)
            y_pred = ensemble.predict(X_test)
            y_proba = ensemble.predict_proba(X_test)

            from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

            ensemble_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc_roc": roc_auc_score(y_test, y_proba),
            }

            logger.info("Ensemble Metrics:")
            for metric_name, value in ensemble_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

            self.training_metrics["ensemble"] = ensemble_metrics

        return self.models

    def save_models(self):
        """Save all trained models and feature extractor."""
        # Save feature extractor
        fe_path = os.path.join(self.model_dir, "feature_extractor.pkl")
        self.feature_extractor.save(fe_path)

        # Save each model
        for model_type, model in self.models.items():
            model_path = os.path.join(self.model_dir, f"{model_type.value}_model.pkl")
            model.save(model_path)

        # Save training metrics and results
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "models": {},
            "optimization_results": {},
            "comparison_results": self.comparison_results,
        }

        for model_type, model in self.models.items():
            results["models"][model_type.value] = {
                "params": model.params,
                "metrics": model.metrics,
            }

        for model_type, opt_result in self.optimization_results.items():
            results["optimization_results"][model_type.value] = opt_result.to_dict()

        results_path = os.path.join(self.model_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nAll models and results saved to {self.model_dir}")

    def load_models(self) -> Dict[ModelType, DDIModel]:
        """Load all saved models."""
        # Load feature extractor
        fe_path = os.path.join(self.model_dir, "feature_extractor.pkl")
        if os.path.exists(fe_path):
            self.feature_extractor = DrugFeatureExtractor.load(fe_path)

        # Load models
        for model_type in ModelType:
            model_path = os.path.join(self.model_dir, f"{model_type.value}_model.pkl")
            if os.path.exists(model_path):
                self.models[model_type] = DDIModel.load(model_path)
                logger.info(f"Loaded {model_type.value} model")

        return self.models

    def get_training_summary(self) -> Dict:
        """Get a summary of training results."""
        summary = {
            "models_trained": len(self.models),
            "model_metrics": {},
            "best_model": None,
            "best_auc": 0,
        }

        for model_type, model in self.models.items():
            summary["model_metrics"][model_type.value] = model.metrics

            if model.metrics.get("auc_roc", 0) > summary["best_auc"]:
                summary["best_auc"] = model.metrics["auc_roc"]
                summary["best_model"] = model_type.value

        if self.comparison_results:
            summary["optimization_comparison"] = self.comparison_results

        return summary


async def train_from_database(
    db_session,
    model_dir: str = "./models",
    n_trials: int = 50,
    run_comparison: bool = True,
) -> Dict:
    """
    Train models using data from the database.

    Args:
        db_session: Async SQLAlchemy session
        model_dir: Directory to save models
        n_trials: Number of optimization trials
        run_comparison: Whether to run optimization method comparison

    Returns:
        Training summary
    """
    from sqlalchemy import select
    from app.models import Drug, DrugInteraction

    logger.info("Loading data from database...")

    # Load drugs
    result = await db_session.execute(select(Drug))
    drugs_orm = result.scalars().all()

    drugs = [
        {
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

    # Load interactions
    result = await db_session.execute(select(DrugInteraction).options())
    interactions_orm = result.scalars().all()

    # Need to load drug names for interactions
    drug_id_to_name = {d.id: d.name for d in drugs_orm}

    interactions = [
        {
            "drug1_name": drug_id_to_name.get(i.drug1_id, ""),
            "drug2_name": drug_id_to_name.get(i.drug2_id, ""),
            "severity": i.severity,
        }
        for i in interactions_orm
    ]

    logger.info(f"Loaded {len(drugs)} drugs and {len(interactions)} interactions")

    # Train
    trainer = DDITrainer(
        model_dir=model_dir, n_trials=n_trials, cv_folds=5, random_state=42
    )

    X_train, X_test, y_train, y_test = trainer.load_data_from_dicts(drugs, interactions)

    trainer.train_all_models(
        X_train, y_train, X_test, y_test, optimize=True, run_comparison=run_comparison
    )

    trainer.save_models()

    return trainer.get_training_summary()
