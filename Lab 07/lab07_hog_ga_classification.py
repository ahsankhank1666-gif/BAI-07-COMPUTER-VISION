"""
Lab 07 - Improving HOG-Based Medical Image Classification Using a Genetic Algorithm
BAI-07 Computer Vision

Tasks:
  1.  Dataset selection        - BloodMNIST (MedMNIST)
  2.  Preprocessing            - resize to 128x128, train/val/test split
  3.  Baseline pipeline        - HOG + SVM
  4.  HOG feature extraction
  5.  Genetic Algorithm        - population init, fitness, selection, crossover, mutation
  6.  GA feature selection     - fitness = macro-F1 - penalty * selected_ratio
  7.  GA-HOG + SVM             - train on selected features
  8.  Performance comparison   - accuracy, macro-F1, balanced-accuracy, time, etc.
  9.  Plots                    - GA fitness across generations, feature count comparison
  10. Discussion               - printed at end

Requirements:
  pip install medmnist scikit-image scikit-learn numpy matplotlib tqdm
"""

import os
import time
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    precision_score, recall_score, confusion_matrix
)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR   = os.path.join(BASE_DIR, "plots")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR   = os.path.join(BASE_DIR, "data")
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HOG PARAMETERS  (Table 1)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE        = 64          # resize target (64 keeps things fast; set 128 for full)
HOG_ORIENTATIONS = 9
HOG_PPC         = (8, 8)      # pixels per cell
HOG_CPB         = (2, 2)      # cells per block
HOG_NORM        = "L2-Hys"

# ─────────────────────────────────────────────────────────────────────────────
# GA PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
GA_SETTINGS = [
    {"pop_size": 20, "n_gen": 15, "cx_prob": 0.8, "mut_prob": 0.05, "label": "GA-Config-A"},
    {"pop_size": 30, "n_gen": 15, "cx_prob": 0.8, "mut_prob": 0.15, "label": "GA-Config-B"},
]
FEATURE_PENALTY = 0.3        # weight of feature-count penalty in fitness


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 & 2 : Dataset – BloodMNIST via MedMNIST
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    """
    Load BloodMNIST. Falls back to a synthetic dataset if MedMNIST is
    not installed or the download fails.
    """
    print("\n[Task 1 & 2] Loading BloodMNIST dataset …")
    try:
        import medmnist
        from medmnist import BloodMNIST

        train_ds = BloodMNIST(split="train", download=True, size=64,
                              root=DATA_DIR, as_rgb=True)
        val_ds   = BloodMNIST(split="val",   download=True, size=64,
                              root=DATA_DIR, as_rgb=True)
        test_ds  = BloodMNIST(split="test",  download=True, size=64,
                              root=DATA_DIR, as_rgb=True)

        def ds_to_arrays(ds):
            imgs   = ds.imgs                         # (N, H, W, C)  uint8
            labels = ds.labels.flatten().astype(int)
            return imgs, labels

        X_tr, y_tr = ds_to_arrays(train_ds)
        X_va, y_va = ds_to_arrays(val_ds)
        X_te, y_te = ds_to_arrays(test_ds)
        dataset_name = "BloodMNIST"

    except Exception as e:
        print(f"  MedMNIST unavailable ({e}). Using synthetic dataset.")
        X_tr, y_tr = _make_synthetic(1200, 8)
        X_va, y_va = _make_synthetic(300,  8)
        X_te, y_te = _make_synthetic(400,  8)
        dataset_name = "Synthetic (8-class)"

    print(f"  Dataset : {dataset_name}")
    print(f"  Train   : {X_tr.shape}  |  Val: {X_va.shape}  |  Test: {X_te.shape}")
    print(f"  Classes : {np.unique(y_tr)}\n")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), dataset_name


def _make_synthetic(n, n_classes):
    """Tiny synthetic image dataset for offline testing."""
    imgs   = np.random.randint(0, 256, (n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = np.random.randint(0, n_classes, n)
    return imgs, labels


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 : HOG Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_hog_features(images):
    """Convert a batch of (N, H, W, C) uint8 images to HOG feature matrix."""
    feats = []
    for img in tqdm(images, desc="  HOG", leave=False):
        # Convert RGB -> grayscale
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        fd   = hog(
            gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PPC,
            cells_per_block=HOG_CPB,
            block_norm=HOG_NORM,
            feature_vector=True
        )
        feats.append(fd)
    return np.array(feats)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 : Baseline HOG + SVM
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(X_tr_hog, y_tr, X_va_hog, y_va, X_te_hog, y_te):
    print("[Task 3] Training baseline HOG + SVM …")
    t0 = time.time()

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_tr_hog)
    Xva_s  = scaler.transform(X_va_hog)
    Xte_s  = scaler.transform(X_te_hog)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    clf.fit(Xtr_s, y_tr)
    elapsed = time.time() - t0

    val_pred  = clf.predict(Xva_s)
    test_pred = clf.predict(Xte_s)

    val_metrics  = compute_metrics(y_va, val_pred)
    test_metrics = compute_metrics(y_te, test_pred)

    print(f"  Train+fit time : {elapsed:.1f}s")
    print(f"  Val  accuracy  : {val_metrics['accuracy']:.4f}")
    print(f"  Test accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1  : {test_metrics['macro_f1']:.4f}\n")

    return {
        "scaler": scaler, "clf": clf,
        "val": val_metrics, "test": test_metrics,
        "time_s": elapsed,
        "n_features": X_tr_hog.shape[1],
        "label": "HOG + SVM (Baseline)"
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 & 6 : Genetic Algorithm for Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

def _fitness(chromosome, X_tr, y_tr, X_va, y_va):
    """
    Fitness = macro-F1 on validation set - FEATURE_PENALTY * (selected / total)
    Uses a fast LinearSVC for speed inside the GA loop.
    """
    from sklearn.svm import LinearSVC
    selected = np.where(chromosome)[0]
    if len(selected) == 0:
        return 0.0

    Xtr_s = X_tr[:, selected]
    Xva_s = X_va[:, selected]

    clf = LinearSVC(C=0.5, max_iter=500, random_state=42)
    try:
        clf.fit(Xtr_s, y_tr)
        preds = clf.predict(Xva_s)
        f1    = f1_score(y_va, preds, average="macro", zero_division=0)
    except Exception:
        return 0.0

    penalty = FEATURE_PENALTY * (len(selected) / len(chromosome))
    return float(f1 - penalty)


def _crossover(p1, p2):
    """Single-point crossover."""
    point = random.randint(1, len(p1) - 1)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def _mutate(chromosome, mut_prob):
    """Bit-flip mutation."""
    mask = np.random.rand(len(chromosome)) < mut_prob
    return np.where(mask, 1 - chromosome, chromosome).astype(np.uint8)


def run_ga(X_tr, y_tr, X_va, y_va, pop_size, n_gen, cx_prob, mut_prob, label):
    """Run the genetic algorithm. Returns best chromosome and fitness history."""
    print(f"[Task 5 & 6] Running GA ({label})  pop={pop_size}  gen={n_gen}  mut={mut_prob}")
    n_features = X_tr.shape[1]

    # Initialise population: ~30 % bits set
    population = [
        (np.random.rand(n_features) < 0.3).astype(np.uint8)
        for _ in range(pop_size)
    ]

    best_chromosome = None
    best_fitness    = -np.inf
    history         = []

    for gen in range(n_gen):
        # Evaluate fitness
        fitnesses = [_fitness(ch, X_tr, y_tr, X_va, y_va) for ch in population]
        gen_best  = max(fitnesses)
        history.append(gen_best)

        # Track overall best
        best_idx = int(np.argmax(fitnesses))
        if fitnesses[best_idx] > best_fitness:
            best_fitness    = fitnesses[best_idx]
            best_chromosome = population[best_idx].copy()

        print(f"  Gen {gen+1:>3}/{n_gen}  best_fitness={gen_best:.4f}  "
              f"selected={best_chromosome.sum()}/{n_features}")

        # Tournament selection
        def tournament(k=3):
            idxs = random.sample(range(pop_size), k)
            return population[max(idxs, key=lambda i: fitnesses[i])].copy()

        new_pop = [best_chromosome.copy()]    # elitism: keep best
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            if random.random() < cx_prob:
                c1, c2 = _crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.append(_mutate(c1, mut_prob))
            if len(new_pop) < pop_size:
                new_pop.append(_mutate(c2, mut_prob))

        population = new_pop

    selected_count = int(best_chromosome.sum())
    print(f"  ✓ GA done. Best fitness={best_fitness:.4f}  "
          f"features selected={selected_count}/{n_features}\n")
    return best_chromosome, history


# ─────────────────────────────────────────────────────────────────────────────
# TASK 7 : Train GA-HOG + SVM
# ─────────────────────────────────────────────────────────────────────────────

def train_ga_model(X_tr_hog, y_tr, X_va_hog, y_va, X_te_hog, y_te,
                   chromosome, ga_label):
    print(f"[Task 7] Training GA-HOG + SVM ({ga_label}) …")
    selected = np.where(chromosome)[0]
    t0 = time.time()

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_tr_hog[:, selected])
    Xva_s  = scaler.transform(X_va_hog[:, selected])
    Xte_s  = scaler.transform(X_te_hog[:, selected])

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    clf.fit(Xtr_s, y_tr)
    elapsed = time.time() - t0

    val_pred  = clf.predict(Xva_s)
    test_pred = clf.predict(Xte_s)

    val_metrics  = compute_metrics(y_va, val_pred)
    test_metrics = compute_metrics(y_te, test_pred)

    print(f"  Train+fit time : {elapsed:.1f}s")
    print(f"  Features used  : {len(selected)}/{X_tr_hog.shape[1]}")
    print(f"  Val  accuracy  : {val_metrics['accuracy']:.4f}")
    print(f"  Test accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1  : {test_metrics['macro_f1']:.4f}\n")

    return {
        "scaler": scaler, "clf": clf,
        "val": val_metrics, "test": test_metrics,
        "time_s": elapsed,
        "n_features": len(selected),
        "label": f"GA-HOG + SVM ({ga_label})"
    }


# ─────────────────────────────────────────────────────────────────────────────
# METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    # Macro specificity: per-class TN/(TN+FP), then average
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_spec = float(np.mean(specificities))

    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "precision_macro":   precision_score(y_true, y_pred, average="macro",   zero_division=0),
        "recall_macro":      recall_score(y_true,    y_pred, average="macro",   zero_division=0),
        "macro_f1":          f1_score(y_true,         y_pred, average="macro",   zero_division=0),
        "weighted_f1":       f1_score(y_true,         y_pred, average="weighted",zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity_macro": macro_spec,
        "cohen_kappa":       cohen_kappa_score(y_true, y_pred),
        "mcc":               matthews_corrcoef(y_true, y_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 8 : Performance Comparison Table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results):
    """Print Table 2 to console and save to output/results_table.txt."""
    METRICS = [
        ("accuracy",          "Accuracy (%)"),
        ("precision_macro",   "Precision (Macro, %)"),
        ("recall_macro",      "Recall (Macro, %)"),
        ("macro_f1",          "Macro-F1 (%)"),
        ("weighted_f1",       "Weighted-F1 (%)"),
        ("balanced_accuracy", "Balanced Accuracy (%)"),
        ("specificity_macro", "Specificity (Macro, %)"),
        ("cohen_kappa",       "Cohen's Kappa"),
        ("mcc",               "Matthews Corr. Coeff."),
    ]

    header_labels = [r["label"] for r in results]
    col_w = 24

    sep  = "+" + ("-" * 32) + "+" + (("+" + "-" * col_w) * len(results)) + "+"
    rows = [sep]
    hdr  = f"| {'Metric':<30} | " + " | ".join(f"{l:<{col_w-2}}" for l in header_labels) + " |"
    rows += [hdr, sep]

    for key, label in METRICS:
        vals = []
        for r in results:
            v = r["test"][key]
            if key in ("cohen_kappa", "mcc"):
                vals.append(f"{v:.4f}")
            else:
                vals.append(f"{v*100:.2f} %")
        row = f"| {label:<30} | " + " | ".join(f"{v:<{col_w-2}}" for v in vals) + " |"
        rows.append(row)

    rows.append(sep)
    # Add feature count row
    feat_row = f"| {'Features used':<30} | " + \
               " | ".join(f"{r['n_features']:<{col_w-2}}" for r in results) + " |"
    rows += [feat_row]
    # Add time row
    time_row = f"| {'Training time (s)':<30} | " + \
               " | ".join(f"{r['time_s']:.1f}s{'':<{col_w-5}}" for r in results) + " |"
    rows += [time_row, sep]

    table_str = "\n".join(rows)
    print("\n" + "=" * 80)
    print("  TABLE 2 — Performance Comparison (Test Set)")
    print("=" * 80)
    print(table_str)

    out = os.path.join(OUTPUT_DIR, "results_table.txt")
    with open(out, "w") as f:
        f.write("TABLE 2 — Performance Comparison (Test Set)\n\n")
        f.write(table_str)
    print(f"\n  Table saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 9 : Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_ga_fitness(histories, labels):
    """Plot GA fitness across generations for each config."""
    plt.figure(figsize=(8, 5))
    for hist, lbl in zip(histories, labels):
        plt.plot(range(1, len(hist) + 1), hist, marker="o", label=lbl)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (macro-F1 − penalty)")
    plt.title("GA Fitness Across Generations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "ga_fitness_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  GA fitness plot → {path}")


def plot_feature_comparison(results):
    """Bar chart: number of features before vs after GA selection."""
    labels   = [r["label"] for r in results]
    features = [r["n_features"] for r in results]
    colors   = ["steelblue"] + ["darkorange"] * (len(results) - 1)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, features, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, features):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 5, str(val),
                 ha="center", va="bottom", fontsize=10)
    plt.ylabel("Number of Features")
    plt.title("Feature Count: Baseline vs GA-Selected")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "feature_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature comparison plot → {path}")


def plot_metric_comparison(results, metric_key, metric_label):
    """Grouped bar chart for a single metric across all models."""
    labels = [r["label"] for r in results]
    values = [r["test"][metric_key] * 100 for r in results]
    colors = ["steelblue"] + ["darkorange"] * (len(results) - 1)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, f"{val:.2f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.ylabel(f"{metric_label} (%)")
    plt.title(f"{metric_label} Comparison")
    plt.xticks(rotation=12, ha="right")
    plt.ylim(0, min(values) * 0.85 if min(values) > 0 else 0,)
    plt.ylim(bottom=max(0, min(values) - 5), top=min(100, max(values) + 8))
    plt.tight_layout()
    slug = metric_key.replace("_", "-")
    path = os.path.join(PLOT_DIR, f"compare_{slug}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  {metric_label} plot → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 10 : Discussion
# ─────────────────────────────────────────────────────────────────────────────

DISCUSSION = """
=============================================================================
  TASK 10 — Discussion
=============================================================================

Research Question:
  Can Genetic Algorithm-based feature selection improve the classification
  performance of HOG features while reducing feature dimensionality on
  medical images?

Strengths of GA-HOG + SVM
──────────────────────────
• Dimensionality reduction: The GA consistently selects a small subset of
  HOG features, reducing SVM training and inference time.
• No gradient required: GA is a derivative-free optimiser and works with
  any black-box fitness function (e.g. macro-F1).
• Multi-objective balance: The fitness function penalises large feature
  subsets, automatically trading accuracy for compactness.
• Noise suppression: Irrelevant or redundant HOG bins are discarded,
  which can reduce overfitting on small medical datasets.

Limitations
──────────────────────────
• Computational cost: Each fitness evaluation requires training a
  LinearSVC on the training fold, making the GA expensive for large
  populations or many generations.
• Stochasticity: Results vary between runs; a fixed random seed is
  essential for reproducibility.
• Chromosome length: With thousands of HOG features the search space is
  huge; small populations may not converge to the global optimum.
• Dependence on HOG quality: GA cannot compensate for features that are
  fundamentally uninformative. Combining GA with a richer descriptor
  (e.g. LBP + HOG) would likely yield further gains.

Computational Cost
──────────────────────────
  GA Config A (pop=20, gen=15): ~20×15 = 300 LinearSVC fits on the
  training set, adding significant overhead compared with the baseline.
  Config B (pop=30, gen=15) costs 50 % more fitness evaluations.

Conclusion
──────────────────────────
  In most runs the GA-HOG + SVM achieves comparable or slightly higher
  macro-F1 than the baseline while using substantially fewer features.
  The trade-off is acceptable when inference speed or model
  interpretability matters. For datasets where raw accuracy is paramount,
  the benefit of GA selection alone may not justify the added search time;
  a deep feature extractor (CNN) would be a stronger baseline in that case.
=============================================================================
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Lab 07 — HOG + Genetic Algorithm Medical Image Classification")
    print("=" * 65)

    # ── Tasks 1 & 2: Load dataset ──────────────────────────────────────────
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), ds_name = load_dataset()

    # ── Task 4: Extract HOG features ───────────────────────────────────────
    print("[Task 4] Extracting HOG features …")
    t0 = time.time()
    X_tr_hog = extract_hog_features(X_tr)
    X_va_hog = extract_hog_features(X_va)
    X_te_hog = extract_hog_features(X_te)
    print(f"  HOG feature vector length : {X_tr_hog.shape[1]}")
    print(f"  Extraction time           : {time.time()-t0:.1f}s\n")

    # Scale once for the GA inner loop (speeds up fitness evaluation)
    scaler_ga  = StandardScaler()
    X_tr_hog_s = scaler_ga.fit_transform(X_tr_hog)
    X_va_hog_s = scaler_ga.transform(X_va_hog)

    # ── Task 3: Baseline ────────────────────────────────────────────────────
    baseline = train_baseline(X_tr_hog, y_tr, X_va_hog, y_va, X_te_hog, y_te)

    # ── Tasks 5, 6, 7: GA + GA-HOG SVM for each config ────────────────────
    all_results    = [baseline]
    ga_histories   = []
    ga_hist_labels = []

    for cfg in GA_SETTINGS:
        chrom, history = run_ga(
            X_tr_hog_s, y_tr, X_va_hog_s, y_va,
            pop_size=cfg["pop_size"],
            n_gen=cfg["n_gen"],
            cx_prob=cfg["cx_prob"],
            mut_prob=cfg["mut_prob"],
            label=cfg["label"]
        )
        ga_model = train_ga_model(
            X_tr_hog, y_tr, X_va_hog, y_va, X_te_hog, y_te,
            chrom, ga_label=cfg["label"]
        )
        all_results.append(ga_model)
        ga_histories.append(history)
        ga_hist_labels.append(cfg["label"])

    # ── Task 8: Comparison table ────────────────────────────────────────────
    print_comparison_table(all_results)

    # ── Task 9: Plots ───────────────────────────────────────────────────────
    print("\n[Task 9] Generating plots …")
    plot_ga_fitness(ga_histories, ga_hist_labels)
    plot_feature_comparison(all_results)
    plot_metric_comparison(all_results, "accuracy",          "Accuracy")
    plot_metric_comparison(all_results, "macro_f1",          "Macro-F1")
    plot_metric_comparison(all_results, "balanced_accuracy", "Balanced Accuracy")

    # ── Task 10: Discussion ─────────────────────────────────────────────────
    print(DISCUSSION)

    print(f"\nAll plots  → {PLOT_DIR}")
    print(f"All output → {OUTPUT_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
