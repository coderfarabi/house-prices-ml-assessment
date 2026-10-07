# House Prices – Advanced Regression Techniques
### Central AI Team | 7-Day Technical Assessment

---

## Overview

A complete end-to-end machine learning project for the [Kaggle House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

**Goal:** Predict the final sale price of residential homes in Ames, Iowa using 79 explanatory features.  
**Metric:** Root Mean Squared Log Error (RMSLE)  
**Deadline:** 8 October 2026

---

## Project Structure

```
house-prices-ml-assessment/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── data/
│   ├── train.csv                    ← Training data (1460 rows, 81 cols)
│   └── test.csv                     ← Test data (1459 rows, 80 cols)
├── notebooks/
│   ├── 01_eda.ipynb                 ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       ← Preprocessing & Feature Engineering
│   ├── 03_model_experiments.ipynb   ← Model comparison experiments
│   └── 04_final_model.ipynb         ← Final model, blending & submission
├── src/
│   ├── data_preprocessing.py        ← Preprocessing pipeline
│   ├── feature_engineering.py       ← Feature engineering functions
│   ├── models.py                    ← Model definitions & training
│   └── evaluation.py                ← Validation & evaluation utilities
├── outputs/
│   ├── figures/                     ← Saved plots
│   ├── model_results.csv            ← CV scores for all models
│   └── submission.csv               ← Kaggle-ready predictions
└── reports/
    └── assessment_report.md         ← Technical report
```

---

## Methodology

### 1. Exploratory Data Analysis
- Target variable distribution analysis (raw vs log-transformed)
- Missing value profiling across all 79 features
- Correlation analysis of numerical features with SalePrice
- Categorical feature distribution and impact on SalePrice
- Outlier detection (Dean De Cock recommendation applied)

### 2. Data Preprocessing
- **Structural NA imputation:** Features like `PoolQC`, `GarageType`, etc. use `'None'` (absence of feature)
- **Numerical NA imputation:** Area/count columns filled with `0`
- **LotFrontage:** Neighbourhood-median imputation
- **Ordinal encoding:** Quality/condition columns (Po=1 → Ex=5)
- **Box-Cox correction:** Applied to skewed numerical features (|skewness| > 0.75, λ=0.15)
- **One-hot encoding:** All remaining categorical columns
- **Rare column dropping:** Columns with < 5 non-dominant values in training set

### 3. Feature Engineering

| Feature | Rationale |
|---------|-----------|
| `TotalSF` | Total habitable area (basement + floors) |
| `TotalBaths` | Weighted bathroom count |
| `TotalPorchSF` | Aggregate outdoor space |
| `HouseAge` | Age at time of sale |
| `RemodAge` | Time since renovation |
| `IsRemodeled` | Whether house was ever remodelled |
| `IsNew` | Brand-new at time of sale |
| `OverallScore` | OverallQual × OverallCond |
| `QualSF` | OverallQual × TotalSF (quality amplifies size value) |
| `QualGrLiv` | OverallQual × GrLivArea |
| `GarageScore` | GarageCars × GarageQual |
| `Has*` flags | Binary presence indicators |

### 4. Models Evaluated

| Model | CV RMSLE | Notes |
|-------|----------|-------|
| Baseline (Ridge) | 0.11316 | Performance floor |
| LASSO | 0.11176 | Sparse linear, auto feature selection |
| Gradient Boosting | 0.11368 | sklearn GBR, Huber loss |
| XGBoost | 0.11161 | Regularised gradient boosting |
| LightGBM | 0.11728 | Leaf-wise boosting |
| **NNLS Ensemble** | **0.10745** | **Final submission** |

### 5. Validation Strategy
- **5-Fold Cross-Validation** with `shuffle=True, random_state=42`
- Out-of-fold (OOF) predictions used for NNLS blending weight estimation
- All preprocessing performed inside the fold to prevent data leakage
- `RobustScaler` fitted only on training folds

### 6. Ensemble Blending
Non-Negative Least Squares (NNLS) is applied to OOF predictions to find optimal non-negative ensemble weights. This automatically down-weights or zeroes out underperforming models.

---

## Setup & Reproduction

### 1. Clone & install dependencies
```bash
git clone <your-repo-url>
cd house-prices-ml-assessment
pip install -r requirements.txt
```

### 2. Data
Download competition data from Kaggle and place in `data/`:
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/
```

### 3. Run notebooks in order
```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_model_experiments.ipynb
notebooks/04_final_model.ipynb
```

### 4. Submission
The final `submission.csv` will be saved to `outputs/submission.csv`.

---

## Results

| Metric | Value |
|--------|-------|
| Best single-model CV RMSLE | 0.11161 |
| Ensemble CV RMSLE | 0.10745 |
| Kaggle Public Score | 0.12314 |
| Kaggle Leaderboard Position | 640 |

---

## Key Findings & Conclusions

- **Log-transforming SalePrice** is essential for RMSLE optimisation — it converts the metric to standard RMSE on the log scale.
- **Domain feature engineering** (especially `TotalSF`, `QualSF`, `HouseAge`) provides significant lift over raw features alone.
- **NNLS ensemble blending** of LASSO + GBR + XGBoost outperforms any individual model by combining complementary linear and non-linear strengths.
- **Residual analysis** shows no systematic bias; main error sources are rare/unique high-value properties and neighbourhood-level pricing nuances.
- **Potential improvements:** Target encoding for Neighbourhood, additional interaction features, stacking with a meta-learner, and Optuna-based hyperparameter tuning.

---

## Author

**Assessment Submission** – Central AI Team Technical Assessment  
**Kaggle Profile:** *[coderfarabi](https://www.kaggle.com/coderfarabi)*  
**Kaggle Submission:** *([submissions](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submissions))*
