#!/usr/bin/env python3
"""
Comparative Implementation of Machine Learning Regression Algorithms
=====================================================================

Same regression problem (Boston house-price prediction), same data, same
features, same target, same 80:20 split - six different regressors:

    1. Linear Regression
    2. Decision Tree Regression
    3. Random Forest Regression
    4. Neural Network Regression (MLP)
    5. XGBoost Regression
    6. K-Nearest Neighbors Regression (KNN)

Every model is evaluated with MAE, MSE, RMSE and R^2 on the untouched test
set. All artifacts (tables, CSVs, figures, configuration) are written to
``results/`` next to this script.

Usage
-----
    python regression_comparison.py
    python regression_comparison.py --data data/boston_housing.csv --seed 42

Author: Muhammad Ahsan Khan (FA23-BAI-052) - BAI-07 Computer Vision
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server-friendly

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "boston_housing.csv"
DEFAULT_RESULTS = HERE / "results"
TARGET = "medv"                     # median owner-occupied home value ($1000s)
TEST_SIZE = 0.20
RANDOM_STATE = 42

MODEL_ORDER = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "Neural Network (MLP)",
    "XGBoost",
    "KNN",
]


def build_models(seed: int = RANDOM_STATE) -> "OrderedDict[str, Pipeline]":
    """Return the six regressors, each wrapped in an identical scaling pipeline.

    A single StandardScaler is fitted on the TRAINING PARTITION ONLY inside
    each pipeline. Using the same pipeline for all six algorithms guarantees
    that every model sees exactly the same input matrix. Tree-based models are
    invariant to positive affine rescaling, so standardisation does not change
    what they learn; it is applied uniformly for a fully fair comparison.
    """
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=3000,
            random_state=seed,
            early_stopping=False,
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        ),
        "KNN": KNeighborsRegressor(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2,
        ),
    }
    return OrderedDict(
        (name, Pipeline([("scaler", StandardScaler()), ("model", est)]))
        for name, est in estimators.items()
    )


# ---------------------------------------------------------------------------
# Data + metrics
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place boston_housing.csv in the data/ folder (see README.md)."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing from {path}")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; impute before training.")
    return df


def split_dataset(df: pd.DataFrame, seed: int):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, shuffle=True
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Four required regression metrics; RMSE is computed explicitly."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    rows, predictions, details = [], {}, {}
    for name, pipe in models.items():
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        test_scores = metrics(y_test, y_pred_test)
        train_scores = metrics(y_train, y_pred_train)

        predictions[name] = np.asarray(y_pred_test, dtype=float)
        rows.append(
            {
                "Model": name,
                **test_scores,
                "Train_MAE": train_scores["MAE"],
                "Train_RMSE": train_scores["RMSE"],
                "Train_R2": train_scores["R2"],
                "Fit_Time_s": float(fit_seconds),
            }
        )

        est = pipe.named_steps["model"]
        info = {
            "hyperparameters": est.get_params(deep=False),
            "fit_time_seconds": float(fit_seconds),
            "warnings_during_fit": sorted({str(w.message) for w in caught}),
        }
        if hasattr(est, "n_iter_"):
            info["n_iter_"] = (
                int(est.n_iter_[0]) if isinstance(est.n_iter_, np.ndarray) else int(est.n_iter_)
            )
        if hasattr(est, "n_features_in_"):
            info["n_features_in_"] = int(est.n_features_in_)
        details[name] = info

        print(
            f"  {name:<22s} "
            f"MAE={test_scores['MAE']:7.4f}  MSE={test_scores['MSE']:8.4f}  "
            f"RMSE={test_scores['RMSE']:7.4f}  R2={test_scores['R2']:7.4f}  "
            f"({fit_seconds:5.2f}s)"
        )
    return pd.DataFrame(rows), predictions, details


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_tables(results_df, out: Path):
    results_df = results_df.copy()
    results_df["RMSE_Rank"] = results_df["RMSE"].rank(method="min").astype(int)
    results_df = results_df.sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(out / "performance_metrics.csv", index=False, float_format="%.6f")
    return results_df


def write_predictions(y_test, predictions, out: Path) -> pd.DataFrame:
    data = {"Test_Index": np.arange(len(y_test)), "Actual": np.asarray(y_test)}
    for name in MODEL_ORDER:
        data[name] = predictions[name]
    pred_df = pd.DataFrame(data)
    pred_df.to_csv(out / "predictions_all_models.csv", index=False, float_format="%.6f")
    return pred_df


def write_markdown_table(results_df, out: Path):
    header = (
        "| Rank | Model | MAE | MSE | RMSE | R² | Train R² | Fit time (s) |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for i, r in results_df.iterrows():
        lines.append(
            f"| {i + 1} | {r['Model']} | {r['MAE']:.4f} | {r['MSE']:.4f} | "
            f"{r['RMSE']:.4f} | {r['R2']:.4f} | {r['Train_R2']:.4f} | "
            f"{r['Fit_Time_s']:.3f} |"
        )
    (out / "performance_table.md").write_text(
        "# Comparative Performance (test set, 20% hold-out)\n\n"
        "Sorted by RMSE (lower is better). R² is the coefficient of determination.\n\n"
        + header + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_configuration(details, df, X_train, X_test, seed, out: Path):
    config = {
        "dataset": {
            "file": str(DEFAULT_DATA.name),
            "rows": int(len(df)),
            "features": list(df.drop(columns=[TARGET]).columns),
            "target": TARGET,
            "target_unit": "median home value in $1000s",
        },
        "split": {
            "type": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": seed,
            "shuffle": True,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "preprocessing": "StandardScaler fitted on training partition only, inside each Pipeline",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "models": details,
    }
    (out / "model_configuration.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )


def make_plots(results_df, y_test, predictions, out: Path):
    sns.set_theme(style="whitegrid", context="notebook")
    y = np.asarray(y_test, dtype=float)

    # (a) 2x3 grid: actual vs predicted scatter for each model
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        yp = predictions[name]
        ax.scatter(y, yp, s=26, alpha=0.75, edgecolor="none", color="#2b6cb0")
        lim = [min(y.min(), yp.min()) - 2, max(y.max(), yp.max()) + 2]
        ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
        ax.set(xlim=lim, ylim=lim, xlabel="Actual MEDV ($1000s)",
               ylabel="Predicted MEDV ($1000s)", title=name)
        row = results_df[results_df["Model"] == name].iloc[0]
        ax.text(0.04, 0.94, f"MAE={row['MAE']:.2f}\nR²={row['R2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#888", alpha=0.85))
    fig.suptitle("Actual vs Predicted House Prices — Test Set (one panel per algorithm)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted_grid.png", dpi=180)
    plt.close(fig)

    # (b) one overlay plot, sorted by actual price
    order = np.argsort(y)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(y[order], label="Actual", color="black", lw=3, alpha=0.85)
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_ORDER)))
    for name, c in zip(MODEL_ORDER, colors):
        ax.plot(np.asarray(predictions[name])[order], label=name, lw=1.4, alpha=0.85, color=c)
    ax.set(xlabel="Test houses (sorted by actual price)",
           ylabel="MEDV ($1000s)",
           title="Actual vs Predicted — All Six Models on the Same Test Set")
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted_all_models.png", dpi=180)
    plt.close(fig)

    # (c) metric bar charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, metric, better in zip(
        axes.ravel(), ["MAE", "MSE", "RMSE", "R2"], ["lower", "lower", "lower", "higher"]
    ):
        d = results_df.sort_values(metric, ascending=(better == "lower"))
        bars = ax.barh(d["Model"], d[metric],
                       color=["#2f855a" if m == d["Model"].iloc[0] else "#718096"
                              for m in d["Model"]])
        ax.set_title(f"{metric} ({better} is better)")
        ax.invert_yaxis()
        for bar, value in zip(bars, d[metric]):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}",
                    va="center", fontsize=9)
    fig.suptitle("Metric-by-Metric Model Comparison (test set)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "metrics_comparison.png", dpi=180)
    plt.close(fig)

    # (d) residual distributions
    resid = pd.DataFrame({name: predictions[name] - y for name in MODEL_ORDER})
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.boxplot(data=resid, ax=ax, palette="vlag", width=0.6)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set(ylabel="Prediction error (predicted − actual), $1000s",
           title="Residual Distribution of Each Model (test set)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "residual_distribution.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the CSV dataset")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Output directory")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")
    args = parser.parse_args(argv)

    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 88)
    print("COMPARATIVE IMPLEMENTATION OF ML REGRESSION ALGORITHMS")
    print("=" * 88)

    df = load_dataset(args.data)
    X_train, X_test, y_train, y_test = split_dataset(df, args.seed)
    print(f"\nDataset   : {args.data.name}  ({len(df)} rows x {df.shape[1]} columns)")
    print(f"Features  : {', '.join(df.drop(columns=[TARGET]).columns)}")
    print(f"Target    : {TARGET} (median home value in $1000s)")
    print(f"Split     : {len(X_train)} train / {len(X_test)} test "
          f"({1 - TEST_SIZE:.0%}:{TEST_SIZE:.0%}, random_state={args.seed})")
    print(f"Scaling   : StandardScaler fit on training data only (inside each pipeline)")
    print(f"Models    : {len(MODEL_ORDER)} algorithms\n")

    models = build_models(args.seed)
    results_df, predictions, details = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    results_df = save_tables(results_df, out)
    best = results_df.iloc[0]
    print("\n" + "-" * 88)
    print("COMPARATIVE PERFORMANCE TABLE (test set; sorted by RMSE)")
    print("-" * 88)
    print(results_df[["Model", "MAE", "MSE", "RMSE", "R2", "Train_R2", "Fit_Time_s"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBEST MODEL: {best['Model']}  "
          f"(RMSE={best['RMSE']:.4f}, MAE={best['MAE']:.4f}, R²={best['R2']:.4f})")

    write_predictions(y_test, predictions, out)
    write_markdown_table(results_df, out)
    write_configuration(details, df, X_train, X_test, args.seed, out)
    make_plots(results_df, y_test, predictions, out)

    print(f"\nFiles in output directory: {out}")
    for f in sorted(out.iterdir()):
        print(f"  - {f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
