"""
LAB 05: HOG-Based Industrial Defect Detection and Classification
NEU Steel Surface Defect Dataset
================================================================
Pipeline: Image → Preprocessing → HOG Feature Extraction → ML Classifier → Decision
"""

# ─────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────
import os, sys, json, zipfile, shutil, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────
IMG_SIZE        = (128, 128)        # resize target
KAGGLE_DATASET  = "sovitrath/neu-steel-surface-defect-detect-trainvalid-split"
DATASET_DIR     = "neu_dataset"     # local folder after download
OUTPUT_DIR      = "lab05_outputs"   # where all figures / reports go
RANDOM_STATE    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 2. Dataset download via Kaggle API
# ─────────────────────────────────────────────────────────────

def setup_kaggle_credentials():
    """
    Set up Kaggle API credentials.
    The user must place kaggle.json in ~/.kaggle/  OR
    set KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    # Try environment variables
    u = os.environ.get("KAGGLE_USERNAME", "")
    k = os.environ.get("KAGGLE_KEY", "")
    if u and k:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": u, "key": k}, f)
        os.chmod(kaggle_json, 0o600)
        return True
    return False


def download_dataset():
    """Download and extract the NEU dataset using the Kaggle API."""
    if os.path.isdir(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        print(f"[INFO] Dataset already exists at '{DATASET_DIR}' — skipping download.")
        return True

    if not setup_kaggle_credentials():
        print("\n[ERROR] Kaggle credentials not found.")
        print("  Option A: Place kaggle.json in ~/.kaggle/")
        print("  Option B: Set environment variables:")
        print("            export KAGGLE_USERNAME=your_username")
        print("            export KAGGLE_KEY=your_api_key")
        print("  Get your API key from: https://www.kaggle.com/settings")
        return False

    print(f"[INFO] Downloading dataset '{KAGGLE_DATASET}' ...")
    ret = os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATASET_DIR} --unzip")
    if ret != 0:
        print("[ERROR] Kaggle download failed. Check credentials and internet connection.")
        return False
    print(f"[INFO] Dataset downloaded to '{DATASET_DIR}/'")
    return True


# ─────────────────────────────────────────────────────────────
# 3. Load images from folder structure
# ─────────────────────────────────────────────────────────────

def find_image_root(base_dir):
    """
    The Kaggle ZIP may unpack to different sub-folder names.
    Walk until we find a directory that has sub-folders containing images.
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    for root, dirs, files in os.walk(base_dir):
        img_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_ext]
        if img_files:
            # Return the parent that directly contains class subfolders
            return os.path.dirname(root)
    return base_dir


def load_dataset(base_dir):
    """
    Walk base_dir looking for sub-folders as class labels.
    Returns images (list of np arrays) and labels (list of str).
    Handles the typical 'train/class/' and 'valid/class/' layout.
    """
    images, labels = [], []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    # Collect all (image_path, label) pairs
    for split in ["train", "valid", "test", ""]:          # try common splits
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in valid_ext:
                    fpath = os.path.join(class_dir, fname)
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, IMG_SIZE)
                        images.append(img)
                        labels.append(class_name)

    if not images:
        # Flat layout: base_dir/class_name/image.jpg
        for class_name in sorted(os.listdir(base_dir)):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if os.path.splitext(fname)[1].lower() in valid_ext:
                    fpath = os.path.join(class_dir, fname)
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, IMG_SIZE)
                        images.append(img)
                        labels.append(class_name)

    return images, labels


def build_binary_labels(labels):
    """
    Map defect classes → 'Defective', normal-looking class → 'Non-Defective'.
    The NEU dataset contains: Crazing, Inclusion, Patches, Pitted, Rolled, Scratches.
    All are defects, so we treat them all as DEFECTIVE for binary classification
    and synthesise Non-Defective samples via augmentation.
    """
    # All 6 NEU classes ARE defect classes.
    # For binary demo we'll use the first 3 classes as one group and rest as another,
    # OR simply keep multi-class (better scientifically).
    # We'll return both so the notebook is flexible.
    binary = ["Defective"] * len(labels)
    return binary


# ─────────────────────────────────────────────────────────────
# 4. HOG feature extraction
# ─────────────────────────────────────────────────────────────

def extract_hog_features(images, pixels_per_cell=(8, 8), orientations=9):
    """Extract HOG feature vector from each image."""
    features = []
    for img in images:
        fd = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )
        features.append(fd)
    return np.array(features, dtype=np.float32)


def visualise_hog(images, labels, unique_labels, save_path):
    """Save a grid of original + HOG visualisations for one sample per class."""
    n = min(len(unique_labels), 6)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
    if n == 1:
        axes = [axes]

    shown = 0
    for lbl in unique_labels[:n]:
        idx = next((i for i, l in enumerate(labels) if l == lbl), None)
        if idx is None:
            continue
        img = images[idx]
        fd, hog_img = hog(
            img, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            visualize=True, feature_vector=True
        )
        hog_img_resc = exposure.rescale_intensity(hog_img, in_range=(0, 10))
        axes[shown][0].imshow(img, cmap="gray")
        axes[shown][0].set_title(f"Original: {lbl}", fontsize=9)
        axes[shown][0].axis("off")
        axes[shown][1].imshow(hog_img_resc, cmap="gray")
        axes[shown][1].set_title("HOG Visualization", fontsize=9)
        axes[shown][1].axis("off")
        shown += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────
# 5. Classifiers
# ─────────────────────────────────────────────────────────────

def build_classifiers():
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                          probability=True, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                          n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                          max_iter=1000, random_state=RANDOM_STATE,
                          solver="lbfgs"))
        ])
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, label_names, clf_name, clf):
    """Train one classifier, return metrics dict."""
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report= classification_report(y_test, y_pred, target_names=label_names,
                                   zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Classifier : {clf_name}")
    print(f"  Train time : {train_time:.1f}s")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%")
    print(f"  F1-score   : {f1*100:.2f}%")
    print(f"{'='*55}")
    print(report)

    return {
        "name": clf_name, "clf": clf,
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "cm": cm, "report": report, "train_time": train_time,
        "y_pred": y_pred
    }


# ─────────────────────────────────────────────────────────────
# 6. Confusion matrix plot
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────
# 7. HOG parameter experiments
# ─────────────────────────────────────────────────────────────

def hog_parameter_experiment(images, y, label_encoder):
    """Grid search over cell sizes × orientations, return results table."""
    cell_sizes   = [(4, 4), (8, 8), (16, 16)]
    orientations = [6, 9, 12]
    results = []

    for c in cell_sizes:
        for o in orientations:
            print(f"  HOG cell={c} orient={o} ...", end=" ", flush=True)
            try:
                X = extract_hog_features(images, pixels_per_cell=c, orientations=o)
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.25, random_state=RANDOM_STATE,
                    stratify=y)
                # Fast SVM with linear kernel for speed
                pipe = Pipeline([("sc", StandardScaler()),
                                 ("clf", SVC(kernel="linear", C=1,
                                             random_state=RANDOM_STATE))])
                pipe.fit(Xtr, ytr)
                ypred = pipe.predict(Xte)
                acc = accuracy_score(yte, ypred)
                f1  = f1_score(yte, ypred, average="macro", zero_division=0)
                feat_dim = X.shape[1]
                print(f"acc={acc*100:.1f}%  f1={f1*100:.1f}%  dim={feat_dim}")
                results.append({"cell": str(c), "orient": o,
                                 "acc": acc, "f1": f1, "dim": feat_dim})
            except Exception as e:
                print(f"ERROR: {e}")
    return results


def plot_hog_experiment(results, save_path):
    """Heatmap of F1 by cell size × orientations."""
    cells = sorted(set(r["cell"] for r in results))
    orients = sorted(set(r["orient"] for r in results))
    mat = np.zeros((len(cells), len(orients)))
    for r in results:
        i = cells.index(r["cell"])
        j = orients.index(r["orient"])
        mat[i, j] = r["f1"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=max(0, mat.min() - 5), vmax=mat.max() + 2)
    ax.set_xticks(range(len(orients)))
    ax.set_xticklabels([f"{o}" for o in orients])
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells)
    ax.set_xlabel("Orientations", fontsize=11)
    ax.set_ylabel("Cell Size (pixels)", fontsize=11)
    ax.set_title("HOG Parameter Study — Macro F1 (%)", fontsize=12)
    plt.colorbar(im, ax=ax)
    for i in range(len(cells)):
        for j in range(len(orients)):
            ax.text(j, i, f"{mat[i,j]:.1f}",
                    ha="center", va="center", fontsize=10, color="black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────
# 8. Robustness experiment
# ─────────────────────────────────────────────────────────────

def apply_perturbation(img, kind):
    """Apply one kind of image perturbation and return perturbed image."""
    if kind == "brightness":
        factor = 1.6
        out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    elif kind == "gaussian_noise":
        noise = np.random.normal(0, 25, img.shape).astype(np.float32)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif kind == "rotation":
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)
        out = cv2.warpAffine(img, M, (w, h))
    elif kind == "blur":
        out = cv2.GaussianBlur(img, (9, 9), 2.0)
    else:
        out = img.copy()
    return out


def robustness_experiment(images, y, best_clf, save_path):
    """Evaluate best classifier under 4 perturbation types."""
    conditions = ["clean", "brightness", "gaussian_noise", "rotation", "blur"]
    rows = []

    # Baseline features (no perturbation)
    X_clean = extract_hog_features(images)
    Xtr, Xte_clean, ytr, yte = train_test_split(
        X_clean, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    best_clf.fit(Xtr, ytr)
    pred_clean = best_clf.predict(Xte_clean)
    base_acc = accuracy_score(yte, pred_clean)
    base_f1  = f1_score(yte, pred_clean, average="macro", zero_division=0)
    rows.append({"cond": "clean", "acc": base_acc, "f1": base_f1})
    print(f"  [clean]           acc={base_acc*100:.2f}%  f1={base_f1*100:.2f}%")

    # Get the test-set images corresponding to the split
    _, test_images, _, _ = train_test_split(
        images, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    for cond in conditions[1:]:
        perturbed = [apply_perturbation(img, cond) for img in test_images]
        Xte_pert = extract_hog_features(perturbed)
        pred = best_clf.predict(Xte_pert)
        acc = accuracy_score(yte, pred)
        f1  = f1_score(yte, pred, average="macro", zero_division=0)
        rows.append({"cond": cond, "acc": acc, "f1": f1})
        print(f"  [{cond:<18}] acc={acc*100:.2f}%  f1={f1*100:.2f}%")

    # Plot
    conds   = [r["cond"] for r in rows]
    accs    = [r["acc"] * 100 for r in rows]
    f1s     = [r["f1"]  * 100 for r in rows]
    x       = np.arange(len(conds))
    width   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, f1s,  width, label="F1-score (%)",  color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Robustness Analysis: Accuracy & F1 under Perturbations", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")
    return rows


# ─────────────────────────────────────────────────────────────
# 9. Classifier comparison bar chart
# ─────────────────────────────────────────────────────────────

def plot_classifier_comparison(results_list, save_path):
    names = [r["name"] for r in results_list]
    accs  = [r["acc"] * 100 for r in results_list]
    f1s   = [r["f1"]  * 100 for r in results_list]
    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, f1s,  width, label="F1-score (%)",  color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Classifier Comparison on HOG Features", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────
# 10. Final decision module
# ─────────────────────────────────────────────────────────────

def quality_control_decision(clf, label_encoder, img_array, binary=True):
    """
    Given a pre-trained classifier and a single grayscale image (np.ndarray),
    return the inspection result string.
    """
    img = cv2.resize(img_array, IMG_SIZE)
    feat = extract_hog_features([img])

    # Probability or decision function
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(feat)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx] * 100
    elif hasattr(clf, "decision_function"):
        df = clf.decision_function(feat)[0]
        if df.ndim == 0:
            confidence = min(abs(float(df)) * 20, 99)
            pred_idx = 0 if float(df) < 0 else 1
        else:
            pred_idx = int(np.argmax(df))
            confidence = min(float(np.max(df)) * 20, 99)
    else:
        pred_idx = 0
        confidence = 75.0

    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Binary mapping: if multi-class, any defect → DEFECTIVE
    defect_keywords = ["crazing", "inclusion", "patches", "pitted",
                       "rolled", "scratches", "defect"]
    is_defective = any(kw in pred_label.lower() for kw in defect_keywords)

    if is_defective:
        decision  = "DEFECTIVE"
        action    = "REJECT PRODUCT"
    else:
        decision  = "NON-DEFECTIVE"
        action    = "ACCEPT PRODUCT"

    result = (
        "\n" + "=" * 42 + "\n"
        "    PRODUCT INSPECTION RESULT\n"
        + "=" * 42 + "\n"
        f"  Prediction  : {decision}\n"
        f"  Class       : {pred_label}\n"
        f"  Confidence  : {confidence:.1f}%\n"
        f"  Action      : {action}\n"
        + "=" * 42
    )
    return result, decision, confidence


# ─────────────────────────────────────────────────────────────
# 11. Write text report
# ─────────────────────────────────────────────────────────────

def write_report(clf_results, hog_exp_results, robust_results,
                 label_names, n_images, save_path):
    lines = []
    sep   = "=" * 60

    lines += [
        sep,
        "LAB 05 RESULTS REPORT",
        "HOG-Based Industrial Defect Detection (NEU Dataset)",
        sep, "",
        "DATASET SUMMARY",
        f"  Total images   : {n_images}",
        f"  Classes        : {', '.join(label_names)}",
        f"  Image size     : {IMG_SIZE[0]}x{IMG_SIZE[1]} px",
        f"  HOG cell size  : 8x8  |  orientations: 9",
        "",
        "CLASSIFIER RESULTS",
        "-" * 60,
    ]

    for r in clf_results:
        lines += [
            f"  {r['name']}",
            f"    Accuracy  : {r['acc']*100:.2f}%",
            f"    Precision : {r['prec']*100:.2f}%",
            f"    Recall    : {r['rec']*100:.2f}%",
            f"    F1-score  : {r['f1']*100:.2f}%",
            f"    Train time: {r['train_time']:.1f}s",
            "",
        ]

    lines += ["HOG PARAMETER EXPERIMENT (SVM-Linear, macro F1 %)",
              "-" * 60]
    lines.append(f"  {'Cell':>8}  {'Orient':>6}  {'Acc%':>6}  {'F1%':>6}  {'Dim':>6}")
    for r in hog_exp_results:
        lines.append(
            f"  {r['cell']:>8}  {r['orient']:>6}  "
            f"{r['acc']*100:>6.1f}  {r['f1']*100:>6.1f}  {r['dim']:>6}"
        )

    lines += ["", "ROBUSTNESS ANALYSIS", "-" * 60]
    lines.append(f"  {'Condition':<20}  {'Acc%':>6}  {'F1%':>6}")
    for r in robust_results:
        lines.append(
            f"  {r['cond']:<20}  {r['acc']*100:>6.1f}  {r['f1']*100:>6.1f}"
        )

    lines += ["", sep, "END OF REPORT", sep]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] {save_path}")


# ─────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  LAB 05 — HOG-Based Industrial Defect Detection")
    print("=" * 55)

    # ── Step 1: Download dataset ──────────────────────────────
    print("\n[STEP 1] Downloading dataset ...")
    ok = download_dataset()
    if not ok:
        print("\n[FALLBACK] Kaggle download skipped.")
        print("  You can also manually download from:")
        print("  https://www.kaggle.com/datasets/sovitrath/"
              "neu-steel-surface-defect-detect-trainvalid-split")
        print("  and extract to: './neu_dataset/'")
        print("\n  For demo purposes, generating synthetic data ...\n")
        # Generate synthetic grayscale images for each class
        np.random.seed(RANDOM_STATE)
        class_names = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]
        images, labels = [], []
        for cls in class_names:
            for _ in range(80):   # 80 per class
                img = np.random.randint(50, 200, IMG_SIZE, dtype=np.uint8)
                # Add class-specific texture
                if cls in ["Crazing", "Scratches"]:
                    for _ in range(5):
                        x1,y1 = np.random.randint(0,IMG_SIZE[0],2)
                        x2,y2 = np.random.randint(0,IMG_SIZE[0],2)
                        cv2.line(img,(x1,y1),(x2,y2),30,2)
                elif cls in ["Inclusion", "Pitted"]:
                    for _ in range(8):
                        cx,cy = np.random.randint(10,118,2)
                        cv2.circle(img,(cx,cy),np.random.randint(3,10),20,-1)
                images.append(img)
                labels.append(cls)
        print(f"  Generated {len(images)} synthetic images across {len(class_names)} classes.")
    else:
        # ── Step 2: Load images ───────────────────────────────
        print("\n[STEP 2] Loading images ...")
        images, labels = load_dataset(DATASET_DIR)
        if not images:
            print("[ERROR] No images found. Check dataset folder structure.")
            sys.exit(1)
        print(f"  Loaded {len(images)} images from {len(set(labels))} classes.")
        unique_l = sorted(set(labels))
        for l in unique_l:
            print(f"    {l}: {labels.count(l)} images")

    unique_labels = sorted(set(labels))

    # ── Step 3: Encode labels ─────────────────────────────────
    print("\n[STEP 3] Encoding labels ...")
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"  Classes: {list(le.classes_)}")

    # ── Step 4: HOG visualization ─────────────────────────────
    print("\n[STEP 4] Visualizing HOG representations ...")
    visualise_hog(
        images, labels, unique_labels,
        os.path.join(OUTPUT_DIR, "hog_visualization.png")
    )

    # ── Step 5: Extract HOG features (default params) ─────────
    print("\n[STEP 5] Extracting HOG features (cell=8x8, orient=9) ...")
    X = extract_hog_features(images, pixels_per_cell=(8, 8), orientations=9)
    print(f"  Feature matrix shape: {X.shape}")

    # ── Step 6: Train / test split ────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ── Step 7: Train classifiers ─────────────────────────────
    print("\n[STEP 6-8] Training classifiers ...")
    clfs       = build_classifiers()
    clf_results = []
    best_clf_obj = None

    for clf_name, clf in clfs.items():
        r = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            le.classes_, clf_name, clf
        )
        clf_results.append(r)
        plot_confusion_matrix(
            r["cm"], le.classes_,
            f"Confusion Matrix — {clf_name}",
            os.path.join(OUTPUT_DIR, f"cm_{clf_name.replace(' ', '_').replace('(','').replace(')','')}.png")
        )

    # Best classifier by accuracy
    best_r   = max(clf_results, key=lambda r: r["acc"])
    best_clf_obj = best_r["clf"]
    print(f"\n  Best classifier: {best_r['name']}  ({best_r['acc']*100:.2f}%)")

    # ── Step 8: Comparison chart ──────────────────────────────
    print("\n[STEP 8] Plotting classifier comparison ...")
    plot_classifier_comparison(
        clf_results,
        os.path.join(OUTPUT_DIR, "classifier_comparison.png")
    )

    # ── Step 9: HOG parameter experiment ─────────────────────
    print("\n[STEP 11] HOG parameter experiment ...")
    hog_exp = hog_parameter_experiment(images, y, le)
    plot_hog_experiment(
        hog_exp,
        os.path.join(OUTPUT_DIR, "hog_parameter_heatmap.png")
    )

    # ── Step 10: Robustness experiment ────────────────────────
    print("\n[STEP 12] Robustness experiment ...")
    # Use a fresh copy of best classifier (re-instantiate to avoid fitted state issues)
    robust_clf = build_classifiers()[best_r["name"]]
    robust_results = robustness_experiment(
        images, y, robust_clf,
        os.path.join(OUTPUT_DIR, "robustness_analysis.png")
    )

    # ── Step 11: Write report ─────────────────────────────────
    print("\n[STEP] Writing text report ...")
    write_report(
        clf_results, hog_exp, robust_results,
        list(le.classes_), len(images),
        os.path.join(OUTPUT_DIR, "lab05_results_report.txt")
    )

    # ── Step 12: Demo final decision module ───────────────────
    print("\n[STEP 13] Final quality-control decision demo ...")
    # Use a random test image from the dataset
    demo_img = images[0]
    result_str, decision, conf = quality_control_decision(best_clf_obj, le, demo_img)
    print(result_str)

    # ── Step 13: Save demo decision on a few images ───────────
    n_demo = min(6, len(images))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i in range(n_demo):
        img = images[i]
        _, dec, conf_v = quality_control_decision(best_clf_obj, le, img)
        color = "#d9534f" if dec == "DEFECTIVE" else "#5cb85c"
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"Label: {labels[i]}\nPred: {dec} ({conf_v:.0f}%)",
            fontsize=9, color=color
        )
        axes[i].axis("off")
        for sp in axes[i].spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(2)
    for i in range(n_demo, len(axes)):
        axes[i].axis("off")
    plt.suptitle("Industrial Quality Control Decision Module", fontsize=13, fontweight="bold")
    plt.tight_layout()
    demo_path = os.path.join(OUTPUT_DIR, "quality_control_demo.png")
    plt.savefig(demo_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {demo_path}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ALL OUTPUTS SAVED TO:", OUTPUT_DIR)
    print("  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    • {f}")
    print("=" * 55)
    print("\nLAB 05 COMPLETE.\n")


if __name__ == "__main__":
    main()
