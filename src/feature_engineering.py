"""
feature_engineering.py
──────────────────────
Domain-informed feature engineering for the House Prices ML Assessment.
Creates aggregated, interaction, and binary indicator features.

Usage
-----
from src.feature_engineering import add_features

df_enriched = add_features(df)
"""

import numpy as np
import pandas as pd

# Ordinal map needed for GarageScore
QUAL_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}


def add_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate area-based features.

    TotalSF      : Total habitable floor area (basement + 1st + 2nd floor).
    TotalBaths   : Weighted bathroom count (half-baths count 0.5).
    TotalPorchSF : Combined porch / deck area.
    """
    df = df.copy()
    df['TotalSF']      = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBaths']   = (
        df['FullBath'] + 0.5 * df['HalfBath']
        + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )
    df['TotalPorchSF'] = (
        df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch']
        + df['3SsnPorch'] + df['ScreenPorch']
    )
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features capturing house age and renovation recency.

    HouseAge    : Years between build and sale.
    RemodAge    : Years between last remodel and sale.
    IsRemodeled : Binary flag – house was ever remodelled.
    IsNew       : Binary flag – sold the same year it was built.
    """
    df = df.copy()
    df['HouseAge']    = (df['YrSold'] - df['YearBuilt']).clip(lower=0)
    df['RemodAge']    = (df['YrSold'] - df['YearRemodAdd']).clip(lower=0)
    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['IsNew']       = (df['YrSold'] == df['YearBuilt']).astype(int)
    return df


def add_quality_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quality × area interactions.  High quality amplifies the value of space.

    OverallScore : OverallQual × OverallCond
    QualSF       : OverallQual × TotalSF
    QualGrLiv    : OverallQual × GrLivArea
    GarageScore  : GarageCars × GarageQual (mapped to integer)
    """
    df = df.copy()
    df['OverallScore'] = df['OverallQual'] * df['OverallCond']

    if 'TotalSF' not in df.columns:
        df = add_area_features(df)   # ensure TotalSF exists

    df['QualSF']    = df['OverallQual'] * df['TotalSF']
    df['QualGrLiv'] = df['OverallQual'] * df['GrLivArea']

    # GarageQual may already be ordinal-encoded (int) or still a string
    garage_qual = df['GarageQual']
    if garage_qual.dtype == object:
        garage_qual = garage_qual.map(QUAL_MAP).fillna(0)
    df['GarageScore'] = df['GarageCars'] * garage_qual

    return df


def add_binary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary presence/absence flags for amenities.
    Presence alone often impacts price independently of size.
    """
    df = df.copy()
    indicator_map = {
        'PoolArea':    'HasPool',
        '2ndFlrSF':    'Has2ndFlr',
        'GarageArea':  'HasGarage',
        'TotalBsmtSF': 'HasBsmt',
        'Fireplaces':  'HasFireplace',
    }
    for col, new_col in indicator_map.items():
        if col in df.columns:
            df[new_col] = (df[col] > 0).astype(int)
    return df


def add_features(df: pd.DataFrame, extra: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    Parameters
    ----------
    df    : Combined train+test DataFrame (after imputation, before encoding).
    extra : If True, include quality×area interaction features.

    Returns
    -------
    df : DataFrame with new engineered columns appended.
    """
    df = add_area_features(df)
    df = add_age_features(df)
    df = add_binary_indicators(df)
    if extra:
        df = add_quality_interactions(df)
    return df
