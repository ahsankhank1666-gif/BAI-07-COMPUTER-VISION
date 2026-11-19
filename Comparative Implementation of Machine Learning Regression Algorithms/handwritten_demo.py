#!/usr/bin/env python3
"""
Handwritten (by-hand) step-by-step solutions for the six regressors.

Every number printed here is computed from the FIRST rows of the same
boston_housing.csv used by regression_comparison.py. The calculations use
small samples (2-5 houses, 1-2 features) so that each arithmetic step can be
checked with a pen and paper, exactly as required by the handwritten part of
the assignment.

    python handwritten_demo.py

Output: results/handwritten_step_by_step.txt (and the same text on stdout).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "boston_housing.csv"
OUT = HERE / "results" / "handwritten_step_by_step.txt"

_lines: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    _lines.append(text)


def section(title: str) -> None:
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def table(headers, rows) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    emit("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    emit("  ".join("-" * w for w in widths))
    for r in rows:
        emit("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def best_split(x: np.ndarray, y: np.ndarray):
    """Exhaustive variance-reduction split for a regression stump (CART)."""
    levels = sorted(set(x))
    thresholds = [(a + b) / 2 for a, b in zip(levels, levels[1:])]
    parent_var = float(np.var(y))
    best = None
    for s in thresholds:
        left, right = y[x <= s], y[x > s]
        if len(left) == 0 or len(right) == 0:
            continue
        weighted = (len(left) / len(y) * float(np.var(left))
                    + len(right) / len(y) * float(np.var(right)))
        gain = parent_var - weighted
        if best is None or gain > best[0]:
            best = (gain, s, float(np.mean(left)), float(np.mean(right)))
    return parent_var, best


def main() -> None:
    df = pd.read_csv(DATA)
    x_rm = df["rm"].to_numpy()[:5]
    x_lstat = df["lstat"].to_numpy()[:5]
    y = df["medv"].to_numpy()[:5]
    names = [f"H{i + 1}" for i in range(5)]

    emit("HANDWRITTEN STEP-BY-STEP SOLUTIONS - SMALL SAMPLE FROM THE DATASET")
    emit("Dataset: boston_housing.csv  |  Target y = MEDV (median house value, $1000s)")
    emit("Sample used below: the first five houses of the dataset")
    table(["House", "rm (rooms)", "lstat (%)", "MEDV ($1000s)"],
          [[names[i], f"{x_rm[i]:.3f}", f"{x_lstat[i]:.2f}", f"{y[i]:.1f}"] for i in range(5)])

    # ------------------------------------------------------------------
    section("1. LINEAR REGRESSION - normal equation, one feature, 3 houses")

    x, yy = x_rm[:3], y[:3]
    mx, my = x.mean(), yy.mean()
    emit(f"\nModel: y_hat = b0 + b1*x, with x = rm, using H1-H3")
    emit(f"x  = [{', '.join(f'{v:.3f}' for v in x)}]")
    emit(f"y  = [{', '.join(f'{v:.1f}' for v in yy)}]")
    emit(f"\nStep 1 - sample means:")
    emit(f"  x_bar = ({' + '.join(f'{v:.3f}' for v in x)}) / 3 = {mx:.5f}")
    emit(f"  y_bar = ({' + '.join(f'{v:.1f}' for v in yy)}) / 3 = {my:.5f}")

    dev_x, dev_y = x - mx, yy - my
    emit(f"\nStep 2 - deviations and their products:")
    table(["House", "x_i - x_bar", "y_i - y_bar", "product", "squared x dev"],
          [[names[i], f"{dev_x[i]:+.5f}", f"{dev_y[i]:+.5f}",
            f"{dev_x[i]*dev_y[i]:+.5f}", f"{dev_x[i]**2:.5f}"] for i in range(3)])

    sxy = float(np.sum(dev_x * dev_y))
    sxx = float(np.sum(dev_x ** 2))
    b1 = sxy / sxx
    b0 = my - b1 * mx
    emit(f"\nStep 3 - sums and coefficients:")
    emit(f"  S_xy = sum((x_i-x_bar)(y_i-y_bar)) = {sxy:.5f}")
    emit(f"  S_xx = sum((x_i-x_bar)^2)          = {sxx:.5f}")
    emit(f"  b1 = S_xy / S_xx = {sxy:.5f} / {sxx:.5f} = {b1:.5f}")
    emit(f"  b0 = y_bar - b1*x_bar = {my:.5f} - {b1:.5f}*{mx:.5f} = {b0:.5f}")
    emit(f"  => y_hat = {b0:.5f} + {b1:.5f} * rm")

    preds = b0 + b1 * x
    emit("\nStep 4 - predictions and residuals on the sample:")
    table(["House", "Actual y", "Predicted y_hat", "Residual y-y_hat"],
          [[names[i], f"{yy[i]:.2f}", f"{preds[i]:.4f}", f"{yy[i]-preds[i]:+.4f}"]
           for i in range(3)])
    emit("  (Two fitted coefficients, three points: least squares minimises the SSE.)")

    # multivariate normal equation on 5 houses
    X = np.column_stack([np.ones(5), x_rm[:5], x_lstat[:5]])
    XtX, Xty = X.T @ X, X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    sklearn_beta = np.r_[LinearRegression().fit(X[:, 1:], y).intercept_,
                         LinearRegression().fit(X[:, 1:], y).coef_]
    emit("\nStep 5 - matrix form for 2 features and 5 houses: beta = (X'X)^-1 X'y")
    emit("  Design matrix X (rows = [1, rm, lstat]):")
    for nm, row in zip(names, X):
        emit(f"    {nm}: [{row[0]:.0f}, {row[1]:6.3f}, {row[2]:5.2f}]")
    emit("  X'X = " + " ; ".join("[" + ", ".join(f"{v:.3f}" for v in r) + "]" for r in XtX))
    emit("  X'y = [" + ", ".join(f"{v:.3f}" for v in Xty) + "]")
    emit(f"  beta (b0, b_rm, b_lstat) = ({beta[0]:.4f}, {beta[1]:.4f}, {beta[2]:.4f})")
    emit(f"  sklearn check           = ({sklearn_beta[0]:.4f}, {sklearn_beta[1]:.4f}, {sklearn_beta[2]:.4f})  -> identical")
    emit("  Example: H1 prediction = b0 + b_rm*6.575 + b_lstat*4.98 = "
         f"{beta @ X[0]:.4f} (actual {y[0]:.1f})")

    # ------------------------------------------------------------------
    section("2. DECISION TREE REGRESSION - CART variance reduction, 4 houses")
    tx, ty = x_lstat[:4], y[:4]
    parent_var, split = best_split(tx, ty)
    gain, s, left_mean, right_mean = split
    left, right = ty[tx <= s], ty[tx > s]
    weighted_var = (len(left) / len(ty)) * float(np.var(left)) + (len(right) / len(ty)) * float(np.var(right))
    emit(f"\nFeature = lstat; candidate split s = {s:.3f} (midpoint of "
         f"{sorted(set(tx))[0]:.2f} and {sorted(set(tx))[1]:.2f})")
    emit(f"Left  (lstat <= {s:.3f}): {[names[i] for i in range(4) if tx[i] <= s]}  y = {left.tolist()}")
    emit(f"Right (lstat >  {s:.3f}): {[names[i] for i in range(4) if tx[i] > s]}  y = {right.tolist()}")

    emit("\nStep 1 - parent impurity (MSE of a constant predictor):")
    emit(f"  y_bar = ({' + '.join(f'{v:.1f}' for v in ty)}) / 4 = {ty.mean():.4f}")
    emit("  MSE_parent = (" + " + ".join(f"({v:.1f}-{ty.mean():.4f})^2" for v in ty)
         + f") / 4 = {parent_var:.4f}")

    emit("\nStep 2 - child impurities:")
    emit(f"  left mean  = ({' + '.join(f'{v:.1f}' for v in left)})/{len(left)} = {left_mean:.4f}")
    emit("  left MSE   = (" + " + ".join(f"({v:.1f}-{left_mean:.4f})^2" for v in left)
         + f")/{len(left)} = {float(np.var(left)):.4f}")
    emit(f"  right mean = ({' + '.join(f'{v:.1f}' for v in right)})/{len(right)} = {right_mean:.4f}")
    emit("  right MSE  = (" + " + ".join(f"({v:.1f}-{right_mean:.4f})^2" for v in right)
         + f")/{len(right)} = {float(np.var(right)):.4f}")
    emit(f"  weighted child MSE = {len(left)}/4*{float(np.var(left)):.4f} "
         f"+ {len(right)}/4*{float(np.var(right)):.4f} = {weighted_var:.4f}")

    emit("\nStep 3 - variance reduction (gain) of the split:")
    emit(f"  gain = MSE_parent - weighted_child_MSE = {parent_var:.4f} - {weighted_var:.4f} = {gain:.4f}")
    emit("  The algorithm picks the split with the LARGEST gain; leaf prediction = mean of y in leaf.")

    sk_tree = DecisionTreeRegressor(max_depth=1, random_state=42).fit(tx.reshape(-1, 1), ty)
    emit(f"\nsklearn check: split threshold = {sk_tree.tree_.threshold[0]:.3f}, "
         f"left leaf (node 1) = {sk_tree.tree_.value[1][0][0]:.4f}, "
         f"right leaf (node 2) = {sk_tree.tree_.value[2][0][0]:.4f}  -> matches by hand")

    # ------------------------------------------------------------------
    section("3. RANDOM FOREST REGRESSION - bootstrap + stump averaging, 5 houses")
    # Two illustrative bootstrap resamples WITH REPLACEMENT (size n = 5).
    bootstraps = [np.array([0, 0, 2, 3, 3]), np.array([1, 1, 2, 4, 4])]
    all_preds = []
    for t, idx in enumerate(bootstraps, start=1):
        xb, yb = x_lstat[idx], y[idx]
        _, best_b = best_split(xb, yb)
        emit(f"\nTree {t}: bootstrap indices = {idx.tolist()}  "
             f"-> lstat = {xb.tolist()}, MEDV = {yb.tolist()}")
        oob = sorted(set(range(5)) - set(idx.tolist()))
        emit(f"  out-of-bag (unused) houses = {[names[j] for j in oob]}")
        if best_b is None:
            leaf = float(yb.mean())
            emit(f"  no usable split -> single leaf prediction {leaf:.4f} for every house")
            p = np.full(5, leaf)
        else:
            g_b, s_b, lm_b, rm_b = best_b
            emit(f"  best stump: s={s_b:.3f}, gain={g_b:.4f}, "
                 f"left leaf={lm_b:.4f}, right leaf={rm_b:.4f}")
            p = np.where(x_lstat <= s_b, lm_b, rm_b)
        all_preds.append(p)
        table(["House", "Tree prediction"],
              [[names[i], f"{p[i]:.4f}"] for i in range(5)])

    rf_pred = np.mean(all_preds, axis=0)
    emit("\nForest output = mean of the 2 tree predictions (full model uses 300 trees):")
    table(["House", "Tree 1", "Tree 2", "Forest (average)", "Actual"],
          [[names[i], f"{all_preds[0][i]:.3f}", f"{all_preds[1][i]:.3f}",
            f"{rf_pred[i]:.4f}", f"{y[i]:.1f}"] for i in range(5)])
    emit("  Averaging decorrelated trees reduces variance compared with a single tree.")

    # ------------------------------------------------------------------
    section("4. NEURAL NETWORK REGRESSION - one forward + backward pass, H1")
    xin = np.array([x_rm[0], x_lstat[0]])       # [rm, lstat] for H1
    target = y[0]
    w1 = np.array([0.5, -0.2])                  # input -> hidden weights
    b1n = -1.0                                  # hidden bias
    w2, b2 = 0.8, 1.0                           # hidden -> output weight/bias
    eta = 0.001
    emit(f"\nTiny net: 2 inputs -> 1 ReLU neuron -> 1 linear output")
    emit(f"Initial weights: w1 = {w1.tolist()}, b1 = {b1n:.2f}, w2 = {w2:.2f}, b2 = {b2:.2f}")
    emit(f"Input H1: x = [{xin[0]:.3f}, {xin[1]:.2f}], target = {target:.1f}")

    z = float(w1 @ xin + b1n)
    a = max(0.0, z)
    yhat = w2 * a + b2
    loss = 0.5 * (target - yhat) ** 2
    emit("\nStep 1 - forward pass:")
    emit(f"  z1   = w1.x + b1 = 0.5*{xin[0]:.3f} + (-0.2)*{xin[1]:.2f} + ({b1n:.1f}) = {z:.4f}")
    emit(f"  a1   = ReLU(z1) = max(0, {z:.4f}) = {a:.4f}")
    emit(f"  y_hat = w2*a1 + b2 = {w2}*{a:.4f} + {b2} = {yhat:.4f}")
    emit(f"  L = 1/2*(y - y_hat)^2 = 0.5*({target:.1f} - {yhat:.4f})^2 = {loss:.4f}")

    dL_dyhat = yhat - target
    dL_dw2 = dL_dyhat * a
    dL_db2 = dL_dyhat
    active = 1.0 if z > 0 else 0.0
    dL_dw1 = dL_dyhat * w2 * active * xin
    dL_db1 = dL_dyhat * w2 * active
    emit("\nStep 2 - backpropagation (chain rule):")
    emit(f"  dL/dy_hat = y_hat - y = {dL_dyhat:+.4f}")
    emit(f"  dL/dw2 = dL/dy_hat * a1 = {dL_dw2:+.4f}")
    emit(f"  dL/db2 = dL/dy_hat      = {dL_db2:+.4f}")
    emit(f"  dL/dw1 = dL/dy_hat * w2 * 1[z>0] * x = [{dL_dw1[0]:+.4f}, {dL_dw1[1]:+.4f}]")
    emit(f"  dL/db1 = dL/dy_hat * w2 * 1[z>0] = {dL_db1:+.4f}  (z>0 so neuron is active)")

    w2n, b2n = w2 - eta * dL_dw2, b2 - eta * dL_db2
    w1n, b1n2 = w1 - eta * dL_dw1, b1n - eta * dL_db1
    z2 = float(w1n @ xin + b1n2)
    yhat2 = w2n * max(0.0, z2) + b2n
    emit("\nStep 3 - gradient descent update (eta = 0.001):")
    emit(f"  w2_new = {w2:.4f} - 0.001*({dL_dw2:+.4f}) = {w2n:.6f}")
    emit(f"  b2_new = {b2:.4f} - 0.001*({dL_db2:+.4f}) = {b2n:.6f}")
    emit(f"  w1_new = [{w1n[0]:.6f}, {w1n[1]:.6f}], b1_new = {b1n2:.6f}")
    emit(f"  new prediction = {yhat2:.4f} (moved toward {target:.1f}); "
         f"new loss = {0.5*(target-yhat2)**2:.4f}")
    emit("  Repeat over all samples/epochs with Adam (full model: 64-32 ReLU layers, 314 iterations).")

    # ------------------------------------------------------------------
    section("5. XGBOOST REGRESSION - residual fitting, one boosting round, 4 houses")
    base = float(ty.mean())
    resid = ty - base
    s = 5.005
    Rl, Rr = resid[tx <= s], resid[tx > s]
    Gl, Gr = float(Rl.sum()), float(Rr.sum())
    Hl, Hr, lam, eta = len(Rl), len(Rr), 1.0, 0.3
    leaf_l, leaf_r = Gl / (Hl + lam), Gr / (Hr + lam)
    gain = 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam)
                  - (Gl + Gr) ** 2 / (Hl + Hr + lam))
    f1 = base + eta * np.where(tx <= s, leaf_l, leaf_r)

    emit("\nStep 1 - initial prediction (base score) = mean(y):")
    emit(f"  F0(x) = {base:.4f} for every house")
    emit("\nStep 2 - negative gradients (residuals) r_i = y_i - F0(x_i):")
    table(["House", "Actual y", "Residual r_i"],
          [[names[i], f"{ty[i]:.1f}", f"{resid[i]:+.4f}"] for i in range(4)])
    emit(f"\nStep 3 - fit a depth-1 tree on residuals, split at lstat = {s:.3f}:")
    emit(f"  left:  G_L = ({' + '.join(f'{v:+.4f}' for v in Rl)}) = {Gl:+.4f}, H_L = {Hl}")
    emit(f"  right: G_R = {Gr:+.4f}, H_R = {Hr}")
    emit(f"  leaf_L = G_L/(H_L+lambda) = {Gl:+.4f}/({Hl}+1) = {leaf_l:+.5f}")
    emit(f"  leaf_R = G_R/(H_R+lambda) = {Gr:+.4f}/({Hr}+1) = {leaf_r:+.5f}")
    emit(f"  split gain = 0.5*[G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda)")
    emit(f"                - (G_L+G_R)^2/(H_L+H_R+lambda)] = {gain:.4f}  (positive -> keep split)")
    emit("\nStep 4 - update F1 = F0 + eta*leaf (eta = 0.3):")
    table(["House", "Leaf value", "F1 prediction", "Actual"],
          [[names[i], f"{leaf_l:+.4f}" if tx[i] <= s else f"{leaf_r:+.4f}",
            f"{f1[i]:.4f}", f"{ty[i]:.1f}"] for i in range(4)])
    emit("  Each later tree repeats this on the NEW residuals (full model: 400 trees, "
         "depth 3, eta = 0.05).")

    # ------------------------------------------------------------------
    section("6. K-NEAREST NEIGHBOURS REGRESSION - H1 query, k=3, 2 raw features")
    q = np.array([x_rm[0], x_lstat[0]])
    cand = np.arange(1, 5)
    d2 = ((np.column_stack([x_rm[cand], x_lstat[cand]]) - q) ** 2).sum(axis=1)
    d = np.sqrt(d2)
    order = np.argsort(d)
    k = 3
    rows = []
    for rank, j in enumerate(order, start=1):
        rows.append([f"{rank}", names[cand[j]],
                     f"({x_rm[cand[j]]:.3f},{x_lstat[cand[j]]:.2f})",
                     f"{d2[j]:.4f}", f"{d[j]:.4f}",
                     "YES" if rank <= k else "no", f"{y[cand[j]]:.1f}"])
    emit(f"\nQuery H1 = (rm={q[0]:.3f}, lstat={q[1]:.2f}); "
         f"distance d = sqrt((rm-rm_q)^2 + (lstat-lstat_q)^2)")
    emit("Step 1 - distances to every other house in the sample:")
    table(["Rank", "House", "(rm, lstat)", "d^2", "d", "Among k=3?", "MEDV"], rows)
    nn = order[:k]
    chosen_y = y[cand[nn]]
    emit("\nStep 2 - average the targets of the 3 nearest neighbours:")
    emit("  neighbours = " + ", ".join(f"{names[cand[j]]} (d={d[j]:.4f}, y={y[cand[j]]:.1f})" for j in nn))
    emit(f"  y_hat(H1) = ({' + '.join(f'{v:.1f}' for v in chosen_y)}) / 3 = {chosen_y.mean():.4f}")
    emit(f"  actual H1 price = {y[0]:.1f}; absolute error = {abs(chosen_y.mean() - y[0]):.4f}")
    emit("  NOTE: the full implementation standardises features first - essential when "
         "features such as 'tax' (0-711) and 'nox' (0-1) are combined. Here raw values "
         "are used only to keep the arithmetic readable.")

    # ------------------------------------------------------------------
    section("SUMMARY")
    emit("Each handwritten derivation above uses a 3-5 house subset of the actual dataset.")
    emit("The Python implementation applies the SAME equations (via scikit-learn / XGBoost)")
    emit("to all 404 training houses and reports metrics on the 102 held-out houses.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    emit(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
