# Lab 07 – Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

> **Course**: BAI-07 Computer Vision  
> **Research Question**: Can Genetic Algorithm-based feature selection improve HOG classification performance while reducing feature dimensionality on medical images?

---

## Project Structure

```
Lab 07/
├── lab07_hog_ga_classification.py   # Main script (all 10 tasks)
├── data/                             # BloodMNIST downloaded here automatically
├── plots/                            # Generated charts
│   ├── ga_fitness_history.png
│   ├── feature_comparison.png
│   ├── compare_accuracy.png
│   ├── compare_macro-f1.png
│   └── compare_balanced-accuracy.png
├── output/
│   └── results_table.txt            # Table 2 saved as text
├── results_report.md                # Full experimental report
└── README.md
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install medmnist scikit-image scikit-learn numpy matplotlib tqdm

# 2. Run the full pipeline
python lab07_hog_ga_classification.py
```

The script will:
1. Download **BloodMNIST** automatically (falls back to synthetic data offline)
2. Extract HOG features from all splits
3. Train baseline **HOG + SVM**
4. Run the **Genetic Algorithm** for two configs (Config-A and Config-B)
5. Train **GA-HOG + SVM** using the selected feature subsets
6. Print **Table 2** (performance comparison) to console and save to `output/results_table.txt`
7. Save all **plots** to `plots/`
8. Print the **Discussion** section

---

## Algorithm Pipeline

```
Images (BloodMNIST)
     │
     ▼
HOG Feature Extraction (9 orientations, 8×8 ppc, 2×2 cpb, L2-Hys)
     │
     ├──► Baseline: HOG + SVM (RBF kernel)
     │
     └──► Genetic Algorithm Feature Selection
               │ Population initialisation (30% bits set)
               │ Fitness = Macro-F1 − 0.3 × (selected / total)
               │ Tournament selection → single-point crossover → bit-flip mutation
               │ Elitism: best chromosome carried forward each generation
               ▼
          GA-HOG + SVM (RBF kernel, selected features only)
```

---

## HOG Parameters (Table 1)

| Parameter | Value |
|---|---|
| Image resolution | 64 × 64 (or 128 × 128) |
| HOG orientations | 9 |
| Pixels per cell | 8 × 8 |
| Cells per block | 2 × 2 |
| Block normalisation | L2-Hys |
| Classifier | SVM (RBF kernel) |
| Data split | Fixed seed (train / val / test) |

---

## GA Configurations

| Parameter | Config-A | Config-B |
|---|---|---|
| Population size | 20 | 30 |
| Generations | 15 | 15 |
| Crossover probability | 0.8 | 0.8 |
| Mutation probability | 0.05 | 0.15 |
| Feature penalty weight | 0.3 | 0.3 |

---

## Key Outputs

| File | Description |
|---|---|
| `plots/ga_fitness_history.png` | Fitness curve per GA config across 15 generations |
| `plots/feature_comparison.png` | Bar chart: baseline features vs GA-selected |
| `plots/compare_accuracy.png` | Accuracy: baseline vs GA-Config-A vs Config-B |
| `plots/compare_macro-f1.png` | Macro-F1 comparison |
| `output/results_table.txt` | Full Table 2 with all 9 metrics |

---

## Quick Summary of Results

> Actual values are printed to the console after running the script.  
> See `results_report.md` for the complete experimental report with discussion.

| Model | Features | Accuracy | Macro-F1 | Training Time |
|---|---|---|---|---|
| HOG + SVM (Baseline) | All HOG | — | — | — |
| GA-HOG + SVM (Config-A) | Reduced | — | — | — |
| GA-HOG + SVM (Config-B) | Reduced | — | — | — |

---

## Dataset

- **BloodMNIST** — 8-class blood cell image dataset from [MedMNIST v2](https://medmnist.com/)
- Tsai et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", *Scientific Data*, 2023.
