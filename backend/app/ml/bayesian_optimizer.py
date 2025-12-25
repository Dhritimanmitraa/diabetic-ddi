"""
Bayesian Hyperparameter Optimization for DDI Models.

Implements three optimization methods for comparison:
- Bayesian Optimization (using Optuna with TPE)
- Grid Search
- Random Search
"""
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
import optuna
from optuna.samplers import TPESampler, GridSampler, RandomSampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
import logging
import json
import os

from app.ml.models import ModelType, DDIModelFactory, DDIModel

logger = logging.getLogger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptimizationMethod(str, Enum):
    """Supported optimization methods."""
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"


class OptimizationResult:
    """Container for optimization results."""
    
    def __init__(
        self,
        method: OptimizationMethod,
        model_type: ModelType,
        best_params: Dict,
        best_score: float,
        n_trials: int,
        total_time: float,
        all_trials: Optional[List[Dict]] = None
    ):
        self.method = method
        self.model_type = model_type
        self.best_params = best_params
        self.best_score = best_score
        self.n_trials = n_trials
        self.total_time = total_time
        self.all_trials = all_trials or []
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'method': self.method.value,
            'model_type': self.model_type.value,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': self.n_trials,
            'total_time_seconds': self.total_time,
            'trials_per_second': self.n_trials / self.total_time if self.total_time > 0 else 0,
        }


class BayesianOptimizer:
    """
    Bayesian hyperparameter optimizer using Optuna.
    
    Uses Tree-structured Parzen Estimator (TPE) as the surrogate model.
    """
    
    def __init__(
        self,
        model_type: ModelType,
        n_trials: int = 100,
        cv_folds: int = 5,
        scoring: str = 'roc_auc',
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.study = None
        self.best_model = None
        
    def _get_param_suggestions(
        self, 
        trial: optuna.Trial, 
        param_space: Dict
    ) -> Dict:
        """Generate parameter suggestions from Optuna trial."""
        params = {}
        
        for param_name, param_range in param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                low, high = param_range
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = trial.suggest_int(param_name, low, high)
                elif isinstance(low, float) or isinstance(high, float):
                    # Use log scale for learning rates
                    if 'learning_rate' in param_name or 'lr' in param_name:
                        params[param_name] = trial.suggest_float(param_name, low, high, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, low, high)
            elif isinstance(param_range, list):
                params[param_name] = trial.suggest_categorical(param_name, param_range)
                
        return params
    
    def _create_objective(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        param_space: Dict
    ) -> Callable:
        """Create objective function for optimization."""
        
        def objective(trial: optuna.Trial) -> float:
            # Get parameter suggestions
            params = self._get_param_suggestions(trial, param_space)
            params['random_state'] = self.random_state
            
            # Create and evaluate model
            model = DDIModelFactory.create(self.model_type, params)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            try:
                scores = cross_val_score(
                    model.model if model.model else DDIModelFactory.create(self.model_type, params).model,
                    X, y,
                    cv=cv,
                    scoring=self.scoring,
                    n_jobs=-1
                )
                
                # Handle the case where model hasn't been initialized
                if model.model is None:
                    temp_model = DDIModelFactory.create(self.model_type, params)
                    # Initialize the underlying sklearn model
                    if self.model_type == ModelType.RANDOM_FOREST:
                        from sklearn.ensemble import RandomForestClassifier
                        temp_model.model = RandomForestClassifier(**params)  # type: ignore[assignment]
                    elif self.model_type == ModelType.XGBOOST:
                        import xgboost as xgb
                        temp_model.model = xgb.XGBClassifier(**params)  # type: ignore[assignment]
                    elif self.model_type == ModelType.LIGHTGBM:
                        import lightgbm as lgb
                        temp_model.model = lgb.LGBMClassifier(**params)  # type: ignore[assignment]
                    
                    scores = cross_val_score(
                        temp_model.model,
                        X, y,
                        cv=cv,
                        scoring=self.scoring,
                        n_jobs=-1
                    )
                
                return scores.mean()
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        return objective
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Optional[Dict] = None,
        method: OptimizationMethod = OptimizationMethod.BAYESIAN
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Feature matrix
            y: Labels
            param_space: Custom parameter space (uses default if None)
            method: Optimization method to use
            
        Returns:
            OptimizationResult with best parameters and metrics
        """
        if param_space is None:
            param_space = DDIModelFactory.get_param_space(self.model_type)
        
        logger.info(f"Starting {method.value} optimization for {self.model_type.value}")
        logger.info(f"Trials: {self.n_trials}, CV folds: {self.cv_folds}")
        
        # Select sampler based on method
        if method == OptimizationMethod.BAYESIAN:
            sampler = TPESampler(seed=self.random_state)
        elif method == OptimizationMethod.RANDOM_SEARCH:
            sampler = RandomSampler(seed=self.random_state)
        else:  # Grid search - need to create search grid
            sampler = self._create_grid_sampler(param_space)
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f"{self.model_type.value}_{method.value}"
        )
        
        # Create objective
        objective = self._create_objective(X, y, param_space)
        
        # Run optimization
        start_time = time.time()
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
            n_jobs=1  # Sequential for stability
        )
        total_time = time.time() - start_time
        
        # Extract results
        all_trials = [
            {
                'number': t.number,
                'params': t.params,
                'value': t.value,
                'state': str(t.state),
            }
            for t in self.study.trials
        ]
        
        result = OptimizationResult(
            method=method,
            model_type=self.model_type,
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            n_trials=len(self.study.trials),
            total_time=total_time,
            all_trials=all_trials
        )
        
        logger.info(f"Optimization complete. Best score: {result.best_score:.4f}")
        logger.info(f"Best params: {result.best_params}")
        logger.info(f"Time: {total_time:.2f}s for {result.n_trials} trials")
        
        return result
    
    def _create_grid_sampler(self, param_space: Dict) -> GridSampler:
        """Create a grid sampler with discrete search space."""
        search_grid = {}
        
        for param_name, param_range in param_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                low, high = param_range
                if isinstance(low, int) and isinstance(high, int):
                    # Create 5 evenly spaced values
                    n_points = min(5, high - low + 1)
                    search_grid[param_name] = list(np.linspace(low, high, n_points, dtype=int))
                elif isinstance(low, float) or isinstance(high, float):
                    search_grid[param_name] = list(np.linspace(low, high, 5))
            elif isinstance(param_range, list):
                search_grid[param_name] = param_range
        
        return GridSampler(search_grid, seed=self.random_state)
    
    def get_optimization_history(self) -> List[Dict]:
        """Get the optimization history."""
        if self.study is None:
            return []
        
        return [
            {
                'trial': t.number,
                'value': t.value,
                'params': t.params,
            }
            for t in self.study.trials
        ]
    
    def get_best_model(self, X: np.ndarray, y: np.ndarray) -> DDIModel:
        """Train and return the best model."""
        if self.study is None:
            raise ValueError("Must run optimize() first")
        
        best_params = self.study.best_params.copy()
        best_params['random_state'] = self.random_state
        
        self.best_model = DDIModelFactory.create(self.model_type, best_params)
        self.best_model.train(X, y)
        
        return self.best_model


class OptimizationComparator:
    """Compare different optimization methods."""
    
    def __init__(
        self,
        model_type: ModelType,
        n_trials_bayesian: int = 100,
        n_trials_random: int = 200,
        n_trials_grid: Optional[int] = None,  # Auto-calculated
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_trials_bayesian = n_trials_bayesian
        self.n_trials_random = n_trials_random
        self.n_trials_grid = n_trials_grid
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        
    def run_comparison(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Optional[Dict] = None
    ) -> Dict[OptimizationMethod, OptimizationResult]:
        """
        Run all optimization methods and compare.
        
        Args:
            X: Feature matrix
            y: Labels
            param_space: Parameter search space
            
        Returns:
            Dictionary mapping method to results
        """
        logger.info("=" * 60)
        logger.info(f"Running optimization comparison for {self.model_type.value}")
        logger.info("=" * 60)
        
        # Bayesian optimization
        logger.info("\n[1/3] Running Bayesian Optimization (TPE)...")
        bayesian_opt = BayesianOptimizer(
            self.model_type,
            n_trials=self.n_trials_bayesian,
            cv_folds=self.cv_folds,
            random_state=self.random_state
        )
        self.results[OptimizationMethod.BAYESIAN] = bayesian_opt.optimize(
            X, y, param_space, OptimizationMethod.BAYESIAN
        )
        
        # Random search
        logger.info("\n[2/3] Running Random Search...")
        random_opt = BayesianOptimizer(
            self.model_type,
            n_trials=self.n_trials_random,
            cv_folds=self.cv_folds,
            random_state=self.random_state
        )
        self.results[OptimizationMethod.RANDOM_SEARCH] = random_opt.optimize(
            X, y, param_space, OptimizationMethod.RANDOM_SEARCH
        )
        
        # Grid search (with limited trials)
        logger.info("\n[3/3] Running Grid Search...")
        grid_trials = self.n_trials_grid or min(100, self.n_trials_bayesian)
        grid_opt = BayesianOptimizer(
            self.model_type,
            n_trials=grid_trials,
            cv_folds=self.cv_folds,
            random_state=self.random_state
        )
        self.results[OptimizationMethod.GRID_SEARCH] = grid_opt.optimize(
            X, y, param_space, OptimizationMethod.GRID_SEARCH
        )
        
        return self.results
    
    def get_comparison_summary(self) -> Dict:
        """Get a summary comparison of all methods."""
        if not self.results:
            return {}
        
        summary = {
            'model_type': self.model_type.value,
            'methods': {},
            'winner': None,
            'efficiency_gain': None,
        }
        
        best_score = -1
        best_method = None
        
        for method, result in self.results.items():
            summary['methods'][method.value] = result.to_dict()
            
            if result.best_score > best_score:
                best_score = result.best_score
                best_method = method
        
        summary['winner'] = best_method.value if best_method else None
        
        # Calculate efficiency (trials needed to reach 95% of best score)
        if OptimizationMethod.BAYESIAN in self.results:
            bayesian = self.results[OptimizationMethod.BAYESIAN]
            random = self.results.get(OptimizationMethod.RANDOM_SEARCH)
            
            if random and random.n_trials > 0:
                summary['efficiency_gain'] = {
                    'bayesian_trials': bayesian.n_trials,
                    'random_trials': random.n_trials,
                    'bayesian_time': bayesian.total_time,
                    'random_time': random.total_time,
                    'trial_reduction_percent': (1 - bayesian.n_trials / random.n_trials) * 100,
                    'time_reduction_percent': (1 - bayesian.total_time / random.total_time) * 100 if random.total_time > 0 else 0,
                }
        
        return summary
    
    def save_results(self, filepath: str):
        """Save comparison results to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'comparison_summary': self.get_comparison_summary(),
            'detailed_results': {
                method.value: result.to_dict()
                for method, result in self.results.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Comparison results saved to {filepath}")

