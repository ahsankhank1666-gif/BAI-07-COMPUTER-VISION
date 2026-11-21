"""
Assignment: Comparative Implementation of Machine Learning Regression Algorithms
Student: Muhammad Ahsan Khan (FA23-BAI-052)
Course: BAI-07 Computer Vision / Machine Learning
Dataset: California Housing Price Prediction (House Price Prediction)

This script implements SIX regression algorithms on the SAME dataset,
SAME features, SAME target, and SAME 80:20 train/test split for fair comparison:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Neural Network Regression (MLPRegressor)
5. XGBoost Regression
6. K-Nearest Neighbors Regression (KNN)

Evaluation Metrics: MAE, MSE, RMSE, R² Score
Outputs: Comparative table + Actual vs Predicted plots saved to ./results/

Run: python regression_comparison.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 70)
print("Comparative Implementation of ML Regression Algorithms")
print("Dataset: California Housing (House Price Prediction)")
print("=" * 70)

# ------------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------------
print("\n[1] Loading dataset...")
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="MedHouseValue")

print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {list(X.columns)}")
print(f"Target: MedHouseValue (median house value in $100k units)")
print("\nFirst 5 rows:")
print(pd.concat([X.head(), y.head()], axis=1).to_string(index=False))
print("\nDataset description:")
print(X.describe().T.to_string())

# Check missing values (California Housing has none, but code is shown for completeness)
print(f"\nMissing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in y: {y.isnull().sum()}")

# ------------------------------------------------------------------
# 2. Train/Test Split (80:20) - SAME for all models
# ------------------------------------------------------------------
print("\n[2] Splitting data 80:20 (random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ------------------------------------------------------------------
# 3. Feature Scaling
#    - Scale-sensitive models (Linear, KNN, MLP) use StandardScaler
#    - Tree-based models (Decision Tree, Random Forest, XGBoost) do NOT need scaling
#    Scaler is fitted ONLY on training data to avoid leakage.
# ------------------------------------------------------------------
print("\n[3] Scaling features (StandardScaler fitted on train only)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Define & Train Six Models
# ------------------------------------------------------------------
print("\n[4] Training six regression models...")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    ),
    "XGBoost Regression": xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
    "KNN Regression (k=5)": KNeighborsRegressor(n_neighbors=5),
}

# Which models need scaled input?
scaled_models = {"Linear Regression", "Neural Network (MLP)", "KNN Regression (k=5)"}

predictions = {}
results = []

for name, model in models.items():
    print(f"  -> Training {name}...")
    if name in scaled_models:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append(
        {"Algorithm": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2_Score": r2}
    )
    print(f"     MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

# ------------------------------------------------------------------
# 5. Comparative Performance Table
# ------------------------------------------------------------------
print("\n[5] Comparative Performance Table (sorted by RMSE - lower is better):")
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values("RMSE").reset_index(drop=True)
print(results_df_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Save to CSV
csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
results_df_sorted.to_csv(csv_path, index=False)
print(f"\nSaved metrics table to: {csv_path}")

best_model = results_df_sorted.iloc[0]["Algorithm"]
print(f"\nBest-performing algorithm (lowest RMSE): {best_model}")

# ------------------------------------------------------------------
# 6. Visualizations: Actual vs Predicted
# ------------------------------------------------------------------
print("\n[6] Generating plots...")

# 6a. Individual Actual vs Predicted plots (one per model)
for name, y_pred in predictions.items():
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, s=15, edgecolors="k", linewidths=0.3)
    # Perfect prediction line (y = x)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction (y=x)")
    plt.xlabel("Actual Values (MedHouseValue)")
    plt.ylabel("Predicted Values (MedHouseValue)")
    plt.title(f"Actual vs Predicted — {name}")
    plt.legend()
    plt.tight_layout()
    fname = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    plt.savefig(os.path.join(RESULTS_DIR, f"actual_vs_predicted_{fname}.png"), dpi=150)
    plt.close()

# 6b. Combined 2x3 grid (all six models together)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.4, s=12, edgecolors="k", linewidths=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(name, fontsize=11, fontweight="bold")
fig.suptitle("Actual vs Predicted — All Six Regression Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "actual_vs_predicted_all.png"), dpi=150)
plt.close()
print("  Saved: actual_vs_predicted_all.png + 6 individual plots")

# 6c. Bar chart: RMSE comparison
plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df_sorted)))
bars = plt.bar(results_df_sorted["Algorithm"], results_df_sorted["RMSE"], color=colors, edgecolor="black")
plt.ylabel("RMSE (lower is better)")
plt.title("Model Comparison — RMSE (California Housing)")
plt.xticks(rotation=15, ha="right")
for bar, val in zip(bars, results_df_sorted["RMSE"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=150)
plt.close()
print("  Saved: metrics_comparison.png")

# 6d. Bar chart: R2 comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(results_df_sorted["Algorithm"], results_df_sorted["R2_Score"], color=colors, edgecolor="black")
plt.ylabel("R² Score (higher is better)")
plt.title("Model Comparison — R² Score (California Housing)")
plt.xticks(rotation=15, ha="right")
for bar, val in zip(bars, results_df_sorted["R2_Score"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "r2_comparison.png"), dpi=150)
plt.close()
print("  Saved: r2_comparison.png")

print("\n" + "=" * 70)
print("Done! All results saved to ./results/")
print("=" * 70)
