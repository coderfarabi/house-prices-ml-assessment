# Technical Assessment Report
## House Prices – Advanced Regression Techniques
**Central AI Team | 7-Day Technical Assessment**  
**Submission Deadline:** 8 October 2026

---

## 1. Problem Statement

**Task:** Predict the sale price of residential homes in Ames, Iowa.  
**Metric:** Root Mean Squared Log Error (RMSLE).  
**Dataset:** 1,460 training samples, 1,459 test samples, 79 explanatory features.

The RMSLE metric penalises relative (percentage) errors symmetrically, making it more appropriate for wide-ranging price distributions than plain RMSE. This motivates **log-transforming SalePrice** before training, which converts the problem to minimising standard RMSE on the log scale.

---

## 2. Exploratory Data Analysis

### 2.1 Target Distribution

Raw `SalePrice` is strongly right-skewed (skewness ≈ 1.88). After applying `log1p`, the distribution becomes approximately normal (skewness ≈ 0.12), validating our decision to train on the log-transformed target.

### 2.2 Missing Values

19 features have missing values. Key patterns:
- `PoolQC` (99%), `MiscFeature` (96%), `Alley` (93%) — absence means no such feature exists.
- `LotFrontage` (18%) — spatially correlated with neighbourhood.
- Electrical, MSZoning — rare missings, safely imputed with mode.

### 2.3 Key Numerical Predictors

Top 5 features by |Pearson correlation| with SalePrice:

| Feature | Correlation |
|---------|-------------|
| OverallQual | 0.79 |
| GrLivArea | 0.71 |
| GarageCars | 0.64 |
| GarageArea | 0.62 |
| TotalBsmtSF | 0.61 |

### 2.4 Outliers

Two samples were identified and removed per the dataset author's recommendation:
- Houses with `GrLivArea > 4,000 sq ft` and `SalePrice < $300,000`.
- These represent partial or non-standard sales that would distort the model.

---

## 3. Data Preprocessing

All preprocessing was applied **jointly to train** to prevent data leakage.

### 3.1 Imputation Strategy

| Category | Strategy | Rationale |
|----------|----------|-----------|
| Structural absence (PoolQC, Alley, etc.) | Fill `'None'` | NA = no feature |
| Area/count columns | Fill `0` | No feature = zero area |
| LotFrontage | Neighbourhood median | Spatial correlation |
| Low-NA categoricals | Mode | Minimal information loss |
| Functional | `'Typ'` | Default value per data description |

### 3.2 Ordinal Encoding

Quality and condition features use a consistent 6-level ordinal map:
`None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5`

This preserves the natural ordering and allows models to learn monotonic relationships.

### 3.3 Skewness Correction

Box-Cox(1p) transformation with λ=0.15 was applied to all numerical features with |skewness| > 0.75. This reduced the number of highly-skewed features from N to approximately 0 and improves linear model performance.

### 3.4 One-Hot Encoding

All remaining categorical features were one-hot encoded. Near-constant dummy columns (< 5 non-dominant values in the training set) were dropped to reduce noise.

---

## 4. Feature Engineering

12 domain-informed features were created before encoding:

| Feature | Formula | Motivation |
|---------|---------|-----------|
| `TotalSF` | Basement + 1st + 2nd floor SF | Primary price driver |
| `TotalBaths` | Weighted bath count | Buyer priority |
| `TotalPorchSF` | Sum of all porch areas | Outdoor space value |
| `HouseAge` | YrSold − YearBuilt | Newer → higher price |
| `RemodAge` | YrSold − YearRemodAdd | Recent reno → higher price |
| `IsRemodeled` | Binary: was ever remodelled | Renovation signal |
| `IsNew` | Binary: sold year = build year | New construction premium |
| `OverallScore` | Qual × Cond | Joint quality metric |
| `QualSF` | Qual × TotalSF | Quality amplifies size value |
| `QualGrLiv` | Qual × GrLivArea | Quality-weighted living area |
| `GarageScore` | Cars × GarageQual | Garage quality × capacity |
| `Has*` flags | Binary presence indicators | Amenity value |

**Final feature count after preprocessing:** ~219 features.

---

## 5. Modelling

### 5.1 Validation Strategy

**5-Fold Cross-Validation** with `shuffle=True, random_state=42`.

Rationale:
- With only 1,460 samples, a single train/test split has high variance. K-fold provides robust, low-variance estimates.
- All preprocessing (including `RobustScaler`) is performed inside each fold to prevent leakage.
- Out-of-fold (OOF) predictions are used for ensemble weight estimation — no additional leakage.

### 5.2 Models Evaluated

| Model | CV RMSLE | Key Parameters |
|-------|----------|----------------|
| Baseline (Ridge) | 0.11316 | α=10, RobustScaler |
| LASSO | 0.11176 | α=0.0005, max_iter=50000, RobustScaler |
| Gradient Boosting | 0.11368 | n=1500, lr=0.02, depth=3, subsample=0.7, Huber loss |
| XGBoost | 0.11161 | n=1500, lr=0.02, depth=3, colsample=0.4, α=0.01 |
| LightGBM | 0.11728 | n=1500, lr=0.02, leaves=8, colsample=0.4 |
| **NNLS Ensemble** | **0.10745** | Convex combination of above |

### 5.3 Ensemble Blending (NNLS)

Non-Negative Least Squares applied to OOF predictions:
- Guarantees non-negative weights (interpretable)
- Automatically zeroes out underperforming models
- Computed entirely on OOF predictions → no leakage

**Key insight:** Linear models (LASSO) and tree-based models make complementary errors. The ensemble combines their strengths.

---

## 6. Error Analysis

**Residual diagnostics:**
- Residuals approximately normally distributed around zero — no systematic bias.
- Slight heteroscedasticity at high price ranges (few training examples).

**Top error patterns:**
1. Very high-end properties (> $400k) — limited training coverage.
2. Certain neighbourhoods with high price variance.
3. `OverallQual = 9-10` homes — small subgroup, harder to generalise.

**Improvement opportunities:**
1. Target encoding for `Neighborhood`
2. `Neighborhood × OverallQual` interaction
3. Stacking with a meta-learner (ElasticNet or Ridge on OOF predictions)
4. Optuna-based hyperparameter optimisation

---

## 7. Final Results

| Item | Value |
|------|-------|
| Validation Strategy | 5-Fold CV, OOF RMSLE |
| Final Model | NNLS Ensemble |
| Ensemble OOF RMSLE | 0.10745 |
| Kaggle Public Score | 0.12314 |
| Submission File | `outputs/submission.csv` |

---

## 8. Reproducibility

1. Clone repository and install: `pip install -r requirements.txt`
2. Add `data/train.csv` and `data/test.csv`
3. Run notebooks in order: `01_eda → 02_preprocessing → 03_model_experiments → 04_final_model`
4. `outputs/submission.csv` is the Kaggle-ready file

All random seeds are fixed (`random_state=42`) throughout.

---

## 9. Conclusions

This project demonstrates a complete, reproducible ML pipeline:
- **Principled EDA** identified key predictors and data quality issues.
- **Domain-informed preprocessing** handled missing values correctly without leakage.
- **Feature engineering** created 12 meaningful features grounded in real-estate domain knowledge.
- **Model experimentation** with 4 diverse algorithms provided complementary strengths.
- **NNLS ensemble blending** improved on the best individual model.
- **Residual analysis** confirmed no systematic bias and identified actionable improvements.

The methodology prioritises **technical soundness and reproducibility** over marginal score gains.
