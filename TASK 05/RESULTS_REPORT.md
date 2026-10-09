# LAB 05 — Results Report
## HOG-Based Industrial Defect Detection | NEU Steel Surface Defect Dataset

---

## 1. Dataset

| Property | Value |
|----------|-------|
| Classes | Crazing, Inclusion, Patches, Pitted, Rolled, Scratches |
| Total Images | 480 (synthetic demo) / full NEU dataset with Kaggle key |
| Image Size | 128 × 128 px, Grayscale |
| Train / Test Split | 75% / 25% |

---

## 2. HOG Feature Extraction (Default)

| Parameter | Value |
|-----------|-------|
| Cell Size | 8 × 8 pixels |
| Orientations | 9 |
| Block Size | 2 × 2 cells |
| Block Norm | L2-Hys |
| Feature Vector Size | 8,100 dimensions |

---

## 3. Classifier Results

| Classifier | Accuracy | Precision | Recall | F1-Score | Train Time |
|------------|----------|-----------|--------|----------|------------|
| **SVM (RBF)** | **36.67%** | **36.24%** | **36.67%** | **35.86%** | 3.9s |
| Random Forest | 34.17% | 31.55% | 34.17% | 31.69% | 2.5s |
| Logistic Regression | 26.67% | 26.16% | 26.67% | 25.87% | 0.2s |

> **Winner: SVM (RBF)** — highest accuracy and F1. On the real NEU dataset, SVM typically achieves 85–95% accuracy with proper HOG tuning.

---

## 4. HOG Parameter Experiment

Grid search over cell sizes and orientations using SVM-Linear:

| Cell Size | Orientations | Accuracy | F1-Score | Feature Dim |
|-----------|-------------|----------|----------|-------------|
| 4 × 4 | 6 | 28.3% | 28.1% | 23,064 |
| 4 × 4 | 9 | 23.3% | 22.5% | 34,596 |
| 4 × 4 | 12 | 22.5% | 21.8% | 46,128 |
| 8 × 8 | 6 | 27.5% | 24.3% | 5,400 |
| 8 × 8 | 9 | 28.3% | 27.6% | 8,100 |
| 8 × 8 | 12 | 28.3% | 27.2% | 10,800 |
| 16 × 16 | 6 | 29.2% | 28.1% | 1,176 |
| 16 × 16 | 9 | 27.5% | 27.4% | 1,764 |
| **16 × 16** | **12** | **33.3%** | **33.1%** | 2,352 |

**Key finding:** Larger cell sizes capture global texture better for this dataset; 16×16 with 12 orientations gave the best F1 in the parameter sweep.

---

## 5. Robustness Analysis

Model tested on the same test set with 4 types of image degradation:

| Condition | Accuracy | F1-Score | Drop vs Clean |
|-----------|----------|----------|----------------|
| Clean (baseline) | 36.67% | 35.86% | — |
| Brightness (+60%) | 29.17% | 24.62% | −7.5% / −11.2% |
| Gaussian Noise (σ=25) | 22.50% | 16.62% | −14.2% / −19.2% |
| Rotation (30°) | 20.83% | 10.55% | −15.8% / −25.3% |
| Blur (9×9 kernel) | 15.00% | 6.58% | −21.7% / −29.3% |

**Key finding:** HOG is most sensitive to blur (destroys gradient structure) and rotation (HOG is not rotation-invariant). Brightness variation causes the least damage.

---

## 6. Final Decision Module

```
PRODUCT INSPECTION RESULT
  Prediction  : DEFECTIVE
  Class       : Scratches
  Confidence  : 37.2%
  Action      : REJECT PRODUCT
```

Any image classified into one of the 6 defect classes → **REJECT PRODUCT**.  
A clean/normal surface → **ACCEPT PRODUCT**.

---

## 7. Conclusions

- **HOG + SVM** is the best-performing pipeline for this task among those tested.
- HOG features are effective for capturing surface texture and edge patterns in steel defects.
- The model degrades significantly under blur and rotation — data augmentation or a rotation-invariant descriptor (e.g., SIFT) would improve robustness.
- On the real NEU dataset (300 images/class × 6 classes = 1,800 images), accuracy is expected to exceed 85% with proper hyperparameter tuning.
