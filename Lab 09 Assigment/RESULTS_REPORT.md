# Detailed Results Report — Regression Algorithms

**Date:** 2026-11-21
**Script:** `regression_comparison.py`
**Dataset:** California Housing (20,640 samples × 8 features)
**Split:** 80% train (16,512) / 20% test (4,128), `random_state=42`
**Scaling:** StandardScaler (fit on train only) for Linear, MLP, KNN

## Metrics (sorted by RMSE)

| Algorithm | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| XGBoost Regression | 0.3156 | 0.2245 | 0.4739 | 0.8287 |
| Random Forest Regression | 0.3275 | 0.2554 | 0.5053 | 0.8051 |
| Neural Network (MLP) | 0.3688 | 0.2924 | 0.5407 | 0.7769 |
| KNN Regression (k=5) | 0.4462 | 0.4324 | 0.6576 | 0.6700 |
| Decision Tree Regression | 0.4547 | 0.4952 | 0.7037 | 0.6221 |
| Linear Regression | 0.5332 | 0.5559 | 0.7456 | 0.5758 |

## Key Observations

1. **XGBoost** wins on every metric (lowest MAE/MSE/RMSE, highest R²).
2. Tree ensembles (XGBoost, RF) clearly outperform a single tree and linear model.
3. MLP (single hidden layer, 100 neurons) underperforms the ensembles but beats Linear/DT/KNN.
4. KNN is decent but limited by high-dimensional distance noise.
5. Linear Regression is the weakest — the data is strongly non-linear.
6. All models show regression-to-the-mean at the top of the target range (price cap at 5.0 in the raw data), visible as a vertical cluster at x≈5 in the scatter plots.

## Environment
- Python 3.13.18
- scikit-learn 1.6.1, xgboost 3.4.1, pandas 2.2.3, numpy 2.3.5, matplotlib 3.11.2
