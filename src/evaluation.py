"""
evaluation.py
─────────────
Validation and evaluation utilities for the House Prices ML Assessment.
Provides RMSLE computation, CV scoring, residual analysis, and result saving.

Usage
-----
from src.evaluation import rmsle_cv, residual_analysis, save_model_results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import cross_val_score, KFold


# ── Metric Helpers ────────────────────────────────────────────────────────────

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Log Error.

    Note: assumes y_true and y_pred are already in log space
    (i.e. log1p-transformed).  RMSE on log1p = RMSLE on raw prices.
    """
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def rmsle_cv(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: KFold
) -> tuple:
    """
    Return (mean, std) RMSLE from cross-validation.

    Uses sklearn's neg_root_mean_squared_error scoring on log-transformed y.
    """
    scores = -cross_val_score(
        model, X, y,
        scoring='neg_root_mean_squared_error',
        cv=cv, n_jobs=-1
    )
    return float(scores.mean()), float(scores.std())


# ── Residual Analysis ─────────────────────────────────────────────────────────

def residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_df: pd.DataFrame,
    save_dir: Path = None
) -> pd.DataFrame:
    """
    Compute residuals and produce diagnostic plots.

    Plots
    -----
    1. Residuals vs Predicted
    2. Residual Distribution
    3. Actual vs Predicted

    Parameters
    ----------
    y_true   : True log-transformed target values (OOF)
    y_pred   : Predicted log-transformed values (OOF)
    train_df : Original training DataFrame (for feature-level error analysis)
    save_dir : If provided, plots are saved as PNG files here

    Returns
    -------
    error_df : DataFrame with per-sample error statistics
    """
    residuals     = y_true - y_pred
    abs_residuals = np.abs(residuals)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color='steelblue')
    axes[0].axhline(0, color='red', linewidth=1, linestyle='--')
    axes[0].set_xlabel('Predicted log(SalePrice)')
    axes[0].set_ylabel('Residual (actual − predicted)')
    axes[0].set_title('Residuals vs Predicted')

    # Plot 2: Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='coral')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_title('Residual Distribution')

    # Plot 3: Actual vs Predicted
    axes[2].scatter(y_true, y_pred, alpha=0.4, s=15, color='steelblue')
    axes[2].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1)
    axes[2].set_xlabel('Actual log(SalePrice)')
    axes[2].set_ylabel('Predicted log(SalePrice)')
    axes[2].set_title('Actual vs Predicted')

    plt.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir / 'residual_diagnostics.png'}")

    plt.show()

    error_df = pd.DataFrame({
        'ActualLogPrice':  y_true,
        'PredLogPrice':    y_pred,
        'Residual':        residuals,
        'AbsResidual':     abs_residuals,
        'PctError':        (abs_residuals / np.abs(y_true) * 100).round(2),
    })
    return error_df


def plot_feature_errors(
    error_df: pd.DataFrame,
    train_df: pd.DataFrame,
    save_dir: Path = None
):
    """
    Plot mean absolute residual by OverallQual and Neighborhood.
    """
    error_df = error_df.copy()
    error_df['OverallQual']  = train_df['OverallQual'].values
    error_df['Neighborhood'] = train_df['Neighborhood'].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.boxplot(data=error_df, x='OverallQual', y='AbsResidual', ax=axes[0], color='steelblue')
    axes[0].set_title('Absolute Residual by OverallQual')

    neigh_err = error_df.groupby('Neighborhood')['AbsResidual'].mean().sort_values(ascending=False).head(15)
    neigh_err[::-1].plot(kind='barh', ax=axes[1], color='coral')
    axes[1].set_title('Mean |Residual| by Neighborhood (Top 15)')

    plt.suptitle('Error Patterns by Feature Group', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        fig.savefig(save_dir / 'feature_error_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {save_dir / 'feature_error_analysis.png'}")

    plt.show()


# ── Results Saving ────────────────────────────────────────────────────────────

def save_model_results(
    cv_scores: dict,
    weights_dict: dict,
    blended_rmsle: float,
    baseline_rmsle: float,
    output_path: Path
):
    """
    Save a model results summary CSV to `output_path`.

    Parameters
    ----------
    cv_scores      : {model_name: float CV RMSLE}
    weights_dict   : {model_name: float NNLS weight}
    blended_rmsle  : Final blended OOF RMSLE
    baseline_rmsle : Baseline (Ridge) CV RMSLE
    output_path    : Path to output CSV file
    """
    rows = [{'Model': 'Baseline (Ridge)', 'CV_RMSLE': round(baseline_rmsle, 5),
              'NNLS_Weight': 0.0, 'Notes': 'Performance floor'}]

    for name in cv_scores:
        rows.append({
            'Model':       name.upper(),
            'CV_RMSLE':    round(cv_scores[name], 5),
            'NNLS_Weight': round(weights_dict.get(name, 0.0), 4),
            'Notes':       ''
        })

    rows.append({
        'Model': 'NNLS Ensemble',
        'CV_RMSLE': round(blended_rmsle, 5),
        'NNLS_Weight': 1.0,
        'Notes': 'Final submission model'
    })

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Model results saved to: {output_path}")
    print(df.to_string(index=False))
    return df
