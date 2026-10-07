"""
data_preprocessing.py
─────────────────────
Core preprocessing pipeline for the House Prices ML Assessment.
Handles missing value imputation, ordinal encoding, skewness correction,
one-hot encoding, and rare-column dropping.

Usage
-----
from src.data_preprocessing import preprocess

X_train, y_train, X_test = preprocess(train_df, test_df)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.special import boxcox1p


# ── Ordinal Encoding Maps ─────────────────────────────────────────────────────

QUAL_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
FIN_MAP  = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}


# ── Imputation ────────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-informed missing value imputation to the combined
    train+test DataFrame.

    Strategy
    --------
    - Structural absence features (PoolQC, Alley, etc.) → 'None'
    - Area / count features (GarageArea, BsmtFinSF1, etc.) → 0
    - LotFrontage → neighbourhood median
    - Low-NA categoricals (MSZoning, Electrical, etc.) → mode
    - Functional → 'Typ', Utilities → 'AllPub'
    """
    df = df.copy()

    # Features where NA means the feature does not exist
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'MasVnrType'
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Area / count columns where NA means zero
    zero_cols = [
        'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Garage year: use YearBuilt as proxy
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

    # LotFrontage: median within same neighbourhood
    if 'LotFrontage' in df.columns:
        df['LotFrontage'] = (
            df.groupby('Neighborhood')['LotFrontage']
              .transform(lambda x: x.fillna(x.median()))
        )
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    # Low-missingness categoricals → mode
    mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df['Functional'] = df['Functional'].fillna('Typ')
    df['Utilities']  = df['Utilities'].fillna('AllPub')

    return df


# ── Ordinal Encoding ──────────────────────────────────────────────────────────

def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace quality/condition string categories with ordered integers.
    Preserves natural ordering: None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5
    """
    df = df.copy()

    qual_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'
    ]
    for col in qual_cols:
        if col in df.columns:
            df[col] = df[col].map(QUAL_MAP).fillna(0)

    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].map(
            {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        )
    if 'BsmtFinType1' in df.columns:
        df['BsmtFinType1'] = df['BsmtFinType1'].map(FIN_MAP)
    if 'BsmtFinType2' in df.columns:
        df['BsmtFinType2'] = df['BsmtFinType2'].map(FIN_MAP)
    if 'GarageFinish' in df.columns:
        df['GarageFinish'] = df['GarageFinish'].map({'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].map(
            {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}
        )
    if 'LotShape' in df.columns:
        df['LotShape'] = df['LotShape'].map({'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3})
    if 'LandSlope' in df.columns:
        df['LandSlope'] = df['LandSlope'].map({'Sev': 0, 'Mod': 1, 'Gtl': 2})
    if 'PavedDrive' in df.columns:
        df['PavedDrive'] = df['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
    if 'CentralAir' in df.columns:
        df['CentralAir'] = (df['CentralAir'] == 'Y').astype(int)
    if 'Fence' in df.columns:
        df['Fence'] = df['Fence'].map({'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4})

    return df


# ── Skewness Correction ───────────────────────────────────────────────────────

def correct_skewness(df: pd.DataFrame, skew_thresh: float = 0.75, lam: float = 0.15) -> pd.DataFrame:
    """
    Apply Box-Cox(1p) transformation to numerical columns whose |skewness|
    exceeds `skew_thresh`. Binary columns (nunique <= 2) are skipped.
    """
    df = df.copy()
    num_cols    = df.select_dtypes(include=[np.number]).columns
    skewness    = df[num_cols].apply(lambda s: skew(s.dropna()))
    skewed_cols = skewness[skewness.abs() > skew_thresh].index
    for col in skewed_cols:
        if df[col].nunique() > 2:
            df[col] = boxcox1p(df[col].clip(lower=0), lam)
    return df


# ── One-Hot Encoding ──────────────────────────────────────────────────────────

def one_hot_encode(df: pd.DataFrame, n_train: int) -> pd.DataFrame:
    """
    Apply get_dummies and drop near-constant columns (< 5 non-dominant values
    in the training portion).
    """
    df = pd.get_dummies(df, drop_first=False)
    keep_cols = [
        c for c in df.columns
        if (df[c].iloc[:n_train] != df[c].iloc[:n_train].iloc[0]).sum() > 5
        or df[c].nunique() > 2
    ]
    return df[keep_cols]


# ── Master Pipeline ───────────────────────────────────────────────────────────

def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame,
    skew_thresh: float = 0.75,
    lam: float = 0.15,
    drop_outliers: bool = True
):
    """
    Full preprocessing pipeline applied jointly to train and test
    to prevent data leakage.

    Parameters
    ----------
    train         : Raw training DataFrame (must include 'SalePrice' and 'Id').
    test          : Raw test DataFrame (must include 'Id').
    skew_thresh   : |skewness| threshold for Box-Cox correction.
    lam           : Box-Cox lambda parameter.
    drop_outliers : If True, removes 2 known Dean De Cock outliers.

    Returns
    -------
    X      : Preprocessed training feature matrix.
    y      : log1p-transformed SalePrice array.
    X_test : Preprocessed test feature matrix (same columns as X).
    """
    tr = train.copy()

    # Remove known outliers (large area, anomalously low price)
    if drop_outliers:
        tr = tr[~((tr['GrLivArea'] > 4000) & (tr['SalePrice'] < 300_000))]

    y       = np.log1p(tr['SalePrice'].values)
    n_train = len(tr)

    # Combine train + test for consistent encoding
    df = pd.concat(
        [tr.drop(columns=['SalePrice', 'Id']),
         test.drop(columns=['Id'])],
        ignore_index=True
    )

    df = impute_missing(df)

    # Cast numeric-but-categorical columns before feature engineering
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['MoSold']     = df['MoSold'].astype(str)
    df['YrSold']     = df['YrSold'].astype(str)

    df = encode_ordinals(df)

    # Drop near-zero-variance / irrelevant columns
    df = df.drop(columns=['Utilities', 'Street', 'PoolQC', 'MiscFeature', 'Alley'],
                 errors='ignore')

    df = correct_skewness(df, skew_thresh=skew_thresh, lam=lam)
    df = one_hot_encode(df, n_train)

    X      = df.iloc[:n_train].reset_index(drop=True)
    X_test = df.iloc[n_train:].reset_index(drop=True)

    return X, y, X_test
