from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping

import pandas as pd
from joblib import parallel_backend
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .utils import load_xy, split_dataset_key


def build_models(random_state: int = 42) -> dict[str, ClassifierMixin | Pipeline]:
    """Create baseline estimators used in the training benchmark."""
    return {
        "GLM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            n_estimators=100,
            criterion = "gini", # {"gini", "entropy", "log_loss"}
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    max_iter=300, random_state=random_state,
                    solver="adam", # {"lbfgs", "sgd", "adam"}
                    
                    )
                    ),
            ]
        ),
    }


def build_param_grids() -> dict[str, dict]:
    """
    Return default hyperparameter grids for each model.

    Returns:
        dict[str, dict]: A dictionary containing hyperparameter grids for three models:
            - "GLM": Logistic Regression parameters
                * clf__C: Inverse of regularization strength (float values: 0.1, 1.0, 10.0)
                * clf__class_weight: Class weight strategy (None or "balanced")
                * clf__solver: Optimization algorithm ("lbfgs")
            - "Random Forest": Random Forest Classifier parameters
                * n_estimators: Number of trees in the forest (200, 400)
                * max_depth: Maximum depth of trees (None for unlimited, or 20)
                * min_samples_leaf: Minimum samples required at leaf node (1, 5)
                * class_weight: Class weight strategy (None or "balanced_subsample")
            - "MLP": Neural Network parameters
                * clf__hidden_layer_sizes: Architecture of hidden layers ((64,), (128, 64))
                * clf__alpha: L2 regularization parameter (0.0001, 0.001)
                * clf__learning_rate_init: Initial learning rate (0.001, 0.01)
    """
    """Return default hyperparameter grids for each model."""
    return {
        "GLM": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
            "clf__solver": ["lbfgs"],
        },
        "Random Forest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 5],
            "class_weight": [None, "balanced_subsample"],
        },
        "MLP": {
            "clf__hidden_layer_sizes": [(64,), (128, 64)],
            "clf__alpha": [0.0001, 0.001],
            "clf__learning_rate_init": [0.001, 0.01],
        },
    }


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Mapping[str, ClassifierMixin | Pipeline],
    param_grids: Mapping[str, dict],
    scoring: str = "f1",
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
    backend: str | None = "threading",
) -> tuple[dict[str, ClassifierMixin | Pipeline], pd.DataFrame]:
    """Tune all models with GridSearchCV and return best estimators and summary."""

    def normalize_param_grid(estimator: ClassifierMixin | Pipeline, grid: dict) -> dict:
        """Map bare estimator params to pipeline-style names when possible."""
        available = estimator.get_params(deep=True)
        normalized: dict = {}
        for key, value in grid.items():
            if key in available:
                normalized[key] = value
                continue

            prefixed_key = f"clf__{key}"
            if prefixed_key in available:
                normalized[prefixed_key] = value
                continue

            normalized[key] = value
        return normalized

    trained_models: dict[str, ClassifierMixin | Pipeline] = {}
    tuning_rows: list[dict] = []

    for model_name, estimator in tqdm(models.items(), desc="Tuning models"):
        grid = normalize_param_grid(estimator, param_grids[model_name])
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        # backend_ctx = parallel_backend(backend) if backend else nullcontext()
        # with backend_ctx:
        #     search.fit(X_train, y_train)
            
        search.fit(X_train, y_train)
        trained_models[model_name] = search.best_estimator_
        tuning_rows.append(
            {
                "model": model_name,
                "best_f1_cv": search.best_score_,
                "best_params": str(search.best_params_),
            }
        )

    tuning_table = pd.DataFrame(tuning_rows)
    return trained_models, tuning_table


def evaluate_models_on_datasets(
    trained_models: Mapping[str, ClassifierMixin | Pipeline],
    test_dataset_keys: list[str],
    train_feature_cols: list[str],
    target_col: str,
    data_dir: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Evaluate trained models on a list of test dataset keys."""
    results_tables_by_model: dict[str, pd.DataFrame] = {}

    for model_name, model in tqdm(trained_models.items(), desc="Evaluating models"):
        rows: list[dict] = []
        for dataset_key in test_dataset_keys:
            league, season = split_dataset_key(dataset_key)
            X_test, y_test = load_xy(
                dataset_key=dataset_key,
                target_col=target_col,
                data_dir=data_dir,
            )
            X_test_aligned = X_test.reindex(columns=train_feature_cols, fill_value=0)
            y_pred = model.predict(X_test_aligned)
            rows.append(
                {
                    "model": model_name,
                    "league": league,
                    "season": season,
                    "tested_league_year": dataset_key,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                }
            )

        table = pd.DataFrame(rows).sort_values(["league", "season"]).reset_index(drop=True)
        results_tables_by_model[model_name] = table

    comparison_table = pd.concat(results_tables_by_model.values(), ignore_index=True)
    comparison_table = comparison_table[
        [
            "model",
            "league",
            "season",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tested_league_year",
        ]
    ]
    comparison_table = comparison_table.sort_values(["model", "league", "season"]).reset_index(
        drop=True
    )

    return results_tables_by_model, comparison_table
