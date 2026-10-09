# LAB 05 — HOG-Based Industrial Defect Detection

Automated visual inspection system using **Histogram of Oriented Gradients (HOG)** features and machine learning classifiers on the **NEU Steel Surface Defect Dataset**.

---

## Pipeline

```
Image → Grayscale + Resize → HOG Feature Extraction → ML Classifier → ACCEPT / REJECT
```

---

## Dataset

**NEU Steel Surface Defect Database** — 6 defect classes:
`Crazing`, `Inclusion`, `Patches`, `Pitted`, `Rolled`, `Scratches`

Download link: [Kaggle — NEU Steel Surface Defect](https://www.kaggle.com/datasets/sovitrath/neu-steel-surface-defect-detect-trainvalid-split)

---

## Setup

```bash
pip install scikit-image scikit-learn matplotlib numpy opencv-python kaggle
```

Place your `kaggle.json` in `~/.kaggle/` then run:

```bash
python hog_defect_detection.py
```

> If no Kaggle credentials are found, the script auto-generates synthetic data and runs the full pipeline anyway — no crash.

---

## What the Script Does

| Step | Description |
|------|-------------|
| 1 | Auto-downloads dataset via Kaggle API |
| 2 | Resizes all images to 128×128, converts to grayscale |
| 3 | Extracts HOG features (cell=8×8, orientations=9) |
| 4 | Visualises HOG representations per class |
| 5 | Trains SVM (RBF), Random Forest, Logistic Regression |
| 6 | Generates confusion matrices + classification reports |
| 7 | Runs HOG parameter grid (cell: 4/8/16, orient: 6/9/12) |
| 8 | Robustness test under brightness, noise, rotation, blur |
| 9 | Final ACCEPT / REJECT decision module |

---

## Output Files (in `lab05_outputs/`)

| File | Description |
|------|-------------|
| `hog_visualization.png` | Original vs HOG image per class |
| `cm_SVM_RBF.png` | Confusion matrix — SVM |
| `cm_Random_Forest.png` | Confusion matrix — Random Forest |
| `cm_Logistic_Regression.png` | Confusion matrix — Logistic Regression |
| `classifier_comparison.png` | Bar chart comparing all classifiers |
| `hog_parameter_heatmap.png` | F1 heatmap across cell sizes × orientations |
| `robustness_analysis.png` | Accuracy & F1 under image perturbations |
| `quality_control_demo.png` | Sample ACCEPT/REJECT decisions |
| `lab05_results_report.txt` | Full numerical results |

---

## Sample Decision Output

```
==========================================
    PRODUCT INSPECTION RESULT
==========================================
  Prediction  : DEFECTIVE
  Class       : Scratches
  Confidence  : 37.2%
  Action      : REJECT PRODUCT
==========================================
```

---

## Tech Stack

`Python 3` · `scikit-learn` · `scikit-image` · `OpenCV` · `NumPy` · `Matplotlib` · `Kaggle API`
