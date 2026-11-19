# Comparative Performance (test set, 20% hold-out)

Sorted by RMSE (lower is better). R² is the coefficient of determination.

| Rank | Model | MAE | MSE | RMSE | R² | Train R² | Fit time (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | XGBoost | 1.7081 | 5.4074 | 2.3254 | 0.9263 | 0.9886 | 0.086 |
| 2 | Random Forest | 2.0822 | 9.0315 | 3.0052 | 0.8768 | 0.9780 | 0.461 |
| 3 | Neural Network (MLP) | 2.2701 | 12.3856 | 3.5193 | 0.8311 | 0.8792 | 0.489 |
| 4 | Decision Tree | 2.7245 | 17.8156 | 4.2209 | 0.7571 | 1.0000 | 0.006 |
| 5 | KNN | 2.5922 | 20.6055 | 4.5393 | 0.7190 | 0.8499 | 0.003 |
| 6 | Linear Regression | 3.1891 | 24.2911 | 4.9286 | 0.6688 | 0.7509 | 0.016 |
