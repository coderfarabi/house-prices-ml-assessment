"""
models.py
─────────
Model definitions and training utilities for the House Prices ML Assessment.
Provides a unified interface for all ML models used in the project.

Usage
-----
from src.models import get_models, train_models, predict_ensemble
"""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_predict

import xgboost as xgb
import lightgbm as lgb


# ── Model Factory ─────────────────────────────────────────────────────────────

def get_baseline() -> object:
    """
    Return a simple Ridge regression baseline model (with RobustScaler).
    Used as the performance floor before more complex models.
    """
    return make_pipeline(RobustScaler(), Ridge(alpha=10))


def get_models() -> dict:
    """
    Return the dictionary of all candidate models.

    Models
    ------
    lasso : L1-regularised linear model – automatic feature selection.
    gbr   : sklearn GradientBoostingRegressor – robust, Huber loss.
    xgb   : XGBoost – regularised gradient boosting.
    lgb   : LightGBM – fast leaf-wise boosting.

    Hyperparameter strategy
    -----------------------
    All tree models use: max_depth=3, learning_rate=0.02, n_estimators=1500,
    subsample=0.7, column subsampling=0.4.  This conservative configuration
    strongly regularises the models to avoid overfitting on ~1460 samples.
    """
    return {
        'lasso': make_pipeline(
            RobustScaler(),
            Lasso(alpha=0.0005, max_iter=50_000)
        ),
        'gbr': GradientBoostingRegressor(
            n_estimators=1500, learning_rate=0.02, max_depth=3,
            max_features='sqrt', min_samples_leaf=10, loss='huber',
            subsample=0.7, random_state=42
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=1500, learning_rate=0.02, max_depth=3,
            subsample=0.7, colsample_bytree=0.4, min_child_weight=2,
            reg_alpha=0.01, reg_lambda=1, random_state=42, n_jobs=-1
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.02, num_leaves=8,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.4,
            min_child_samples=10, reg_alpha=0.01, verbose=-1,
            random_state=42, n_jobs=-1
        ),
    }


# ── Cross-Validation ──────────────────────────────────────────────────────────

def get_oof_predictions(
    models: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: KFold
) -> dict:
    """
    Generate out-of-fold (OOF) predictions for each model using cross_val_predict.

    OOF predictions are honest cross-validated predictions — each sample is
    predicted by a model that never saw it during training.  They are used
    for ensemble weight estimation without data leakage.

    Parameters
    ----------
    models : dict of {name: sklearn-compatible model}
    X      : Feature matrix (training set)
    y      : Target vector (log1p SalePrice)
    cv     : KFold splitter

    Returns
    -------
    oof_preds : dict of {name: np.ndarray of OOF predictions}
    """
    oof_preds = {}
    for name, model in models.items():
        oof_preds[name] = cross_val_predict(model, X, y, cv=cv)
        rmsle = np.sqrt(np.mean((oof_preds[name] - y) ** 2))
        print(f"  {name.upper():6s}  OOF RMSLE: {rmsle:.5f}")
    return oof_preds


# ── NNLS Ensemble Blending ────────────────────────────────────────────────────

def compute_nnls_weights(oof_preds: dict, y: np.ndarray) -> dict:
    """
    Compute Non-Negative Least Squares (NNLS) ensemble weights from OOF
    predictions.

    NNLS finds w >= 0 such that P @ w ≈ y in the least-squares sense.
    Weights are normalised to sum to 1 (convex combination).

    Parameters
    ----------
    oof_preds : dict of {name: np.ndarray}
    y         : True log-transformed target

    Returns
    -------
    weights_dict : dict of {name: float weight}
    """
    model_names = list(oof_preds)
    P = np.column_stack([oof_preds[n] for n in model_names])
    weights, _ = nnls(P, y)
    weights /= weights.sum()
    return dict(zip(model_names, weights))


# ── Full Training & Prediction ────────────────────────────────────────────────

def train_models(
    models: dict,
    X: pd.DataFrame,
    y: np.ndarray
) -> dict:
    """
    Retrain all models on the full training set (no CV splits).
    Call after weight estimation to prepare for test-set prediction.

    Returns the same models dict (fitted in-place) for convenience.
    """
    for name, model in models.items():
        model.fit(X, y)
        print(f"  {name.upper():6s}  trained on {len(X)} samples.")
    return models


def predict_ensemble(
    models: dict,
    weights_dict: dict,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Generate weighted ensemble predictions on the test set.

    Parameters
    ----------
    models       : dict of fitted models
    weights_dict : dict of {name: float weight} (from compute_nnls_weights)
    X_test       : Preprocessed test feature matrix

    Returns
    -------
    predictions : np.ndarray of log-scale predictions (apply np.expm1 to get USD)
    """
    predictions = np.zeros(len(X_test))
    for name, model in models.items():
        predictions += weights_dict[name] * model.predict(X_test)
    return predictions
