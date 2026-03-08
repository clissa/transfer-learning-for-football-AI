"""
Smoke test: train baseline model on toy dataset (CPU, <30s, deterministic).
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Optional
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


@pytest.fixture
def toy_dataset():
    """Generate minimal toy dataset (10 games, ~50 actions each)."""
    np.random.seed(42)
    random.seed(42)
    
    n_games = 10
    actions_per_game = 50
    n_features = 20
    
    rows = []
    for game_id in range(n_games):
        for action_id in range(actions_per_game):
            feature_vals = {f'feat_{i}': np.random.randn() for i in range(n_features)}
            row = {
                'game_id': game_id,
                'action_id': action_id,
                'scores': int(np.random.rand() > 0.85),  # ~15% positive
                'concedes': int(np.random.rand() > 0.90),  # ~10% positive
                **feature_vals,
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def test_baseline_model_train_deterministic(toy_dataset):
    """
    Train LogisticRegression on toy data twice with same seed.
    Verify metrics are identical (deterministic).
    """
    seed = 123
    feature_cols = [c for c in toy_dataset.columns if c.startswith('feat_')]
    
    metrics_run1 = _train_and_evaluate(toy_dataset, feature_cols, seed=seed)
    metrics_run2 = _train_and_evaluate(toy_dataset, feature_cols, seed=seed)
    
    # Check both AUC scores match
    assert np.isclose(
        metrics_run1['p_score_auc'],
        metrics_run2['p_score_auc'],
        rtol=1e-6
    ), f"p_score AUC not deterministic: {metrics_run1['p_score_auc']} vs {metrics_run2['p_score_auc']}"
    
    assert np.isclose(
        metrics_run1['p_concede_auc'],
        metrics_run2['p_concede_auc'],
        rtol=1e-6
    ), f"p_concede AUC not deterministic: {metrics_run1['p_concede_auc']} vs {metrics_run2['p_concede_auc']}"


def test_baseline_model_artifacts(toy_dataset, tmp_path):
    """
    Train model and verify artifacts (model, config, metrics) are saved.
    """
    seed = 456
    feature_cols = [c for c in toy_dataset.columns if c.startswith('feat_')]
    
    output_dir = Path(tmp_path) / "smoke_output"
    output_dir.mkdir()
    
    metrics = _train_and_evaluate(
        toy_dataset,
        feature_cols,
        seed=seed,
        output_dir=output_dir,
    )
    
    # Verify metrics were logged
    assert metrics['p_score_auc'] > 0.4, "p_score AUC suspiciously low"
    assert metrics['p_concede_auc'] > 0.3, "p_concede AUC suspiciously low"
    
    # Verify output files exist (if output_dir was used)
    if output_dir:
        metrics_file = output_dir / "metrics.json"
        assert metrics_file.exists(), f"metrics.json not found in {output_dir}"
        
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
        assert saved_metrics['p_score_auc'] == pytest.approx(metrics['p_score_auc'])


def _train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    seed: int = 123,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Train baseline logistic regression on toy data.
    
    Args:
        df: DataFrame with 'game_id', 'action_id', 'scores', 'concedes', + feature columns.
        feature_cols: List of feature column names.
        seed: Random seed (set before train).
        output_dir: If provided, save metrics.json to this directory.
    
    Returns:
        dict with keys: 'p_score_auc', 'p_concede_auc', 'p_score_brier', 'p_concede_brier'
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Simple train/test split by game_id (no temporal leakage within game)
    game_ids = sorted(df['game_id'].unique())
    cut = int(0.8 * len(game_ids))
    train_games = set(game_ids[:cut])
    test_games = set(game_ids[cut:])
    
    train = df[df.game_id.isin(train_games)].copy()
    test = df[df.game_id.isin(test_games)].copy()
    
    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)
    
    y_score_train = train['scores'].astype(int)
    y_score_test = test['scores'].astype(int)
    y_concede_train = train['concedes'].astype(int)
    y_concede_test = test['concedes'].astype(int)
    
    # Train baseline models
    p_score = LogisticRegression(
        max_iter=100,
        random_state=seed,
    )
    p_concede = LogisticRegression(
        max_iter=100,
        random_state=seed,
    )
    
    p_score.fit(X_train, y_score_train)
    p_concede.fit(X_train, y_concede_train)
    
    # Evaluate
    p_score_hat = p_score.predict_proba(X_test)[:, 1]
    p_concede_hat = p_concede.predict_proba(X_test)[:, 1]
    
    metrics = {
        'p_score_auc': roc_auc_score(y_score_test, p_score_hat),
        'p_score_brier': brier_score_loss(y_score_test, p_score_hat),
        'p_concede_auc': roc_auc_score(y_concede_test, p_concede_hat),
        'p_concede_brier': brier_score_loss(y_concede_test, p_concede_hat),
    }
    
    # Save if output_dir is provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = Path(output_dir) / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics