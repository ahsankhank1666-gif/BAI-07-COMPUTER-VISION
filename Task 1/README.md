# Deep Learning Model Benchmarking & Transfer Learning Evaluation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides experimental benchmark evaluations for multiple transfer learning architectures (CNN backbones) and classical classifiers on image classification tasks. It includes structured performance reports, raw evaluation CSV datasets, and formatted documentation ready for academic and professional presentation.

---

## 📊 Results Summary

### Table 1. Comparison of Transfer Learning Models
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) | Train Time (min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| alexnet | 58.75 | 63.38 | 58.75 | 55.22 | 88.30 | 4.0 |
| vgg16 | 61.25 | 69.52 | 61.25 | 54.31 | 88.16 | 5.1 |
| vgg19 | 60.00 | 64.22 | 60.00 | 58.62 | 88.69 | 5.3 |
| resnet18 | 61.25 | 63.04 | 61.25 | 54.84 | 90.72 | 3.7 |
| resnet50 | 57.50 | 58.89 | 57.50 | 54.58 | 87.29 | 4.3 |
| resnet101 | 58.75 | 54.49 | 58.75 | 51.91 | 92.42 | 4.8 |
| densenet121 | 61.25 | 58.23 | 61.25 | 55.70 | 91.41 | 4.4 |
| efficientnet_b0 | 62.50 | 65.65 | 62.50 | 58.08 | 90.84 | 4.0 |

### Table 2. Comparison of Different Classifiers (on EfficientNet-B0 Deep Features)
| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Deep Features | Logistic Regression | 67.50 | 70.90 | 67.50 | 64.16 | 91.70 |
| Deep Features | Decision Tree | 57.50 | 61.27 | 57.50 | 56.31 | 73.44 |
| Deep Features | Random Forest | 58.75 | 64.14 | 58.75 | 52.71 | 90.88 |
| Deep Features | K-Nearest Neighbors (KNN) | 61.25 | 63.14 | 61.25 | 57.35 | 82.71 |
| Deep Features | Linear SVM | 63.75 | 64.40 | 63.75 | 59.30 | 90.92 |
| Deep Features | RBF-SVM | 60.00 | 61.04 | 60.00 | 53.52 | 92.30 |
| Deep Features | XGBoost | 60.00 | 64.71 | 60.00 | 54.32 | 88.11 |

### Table 3. Computational Efficiency Comparison
| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| alexnet | 57.02 | 217.54 | 0.71 | 2.10 | 58.75 |
| vgg16 | 134.28 | 512.25 | 15.47 | 10.68 | 61.25 |
| vgg19 | 139.59 | 532.51 | 19.63 | 12.89 | 60.00 |
| resnet18 | 11.18 | 42.72 | 1.82 | 2.37 | 61.25 |
| resnet50 | 23.52 | 90.02 | 4.13 | 5.62 | 57.50 |
| resnet101 | 42.51 | 162.77 | 7.86 | 11.32 | 58.75 |
| densenet121 | 6.96 | 27.13 | 2.90 | 19.29 | 61.25 |
| efficientnet_b0 | 4.01 | 15.60 | 0.41 | 10.29 | 62.50 |

## 📂 Repository Contents

* **`Task 01 - Updated.docx`**: The updated Word document containing populated benchmark tables.
* **`Task_01_Results.md`**: Markdown-formatted report for direct viewing and documentation on GitHub.
* **`table1_transfer_learning_results.csv`**: Raw experimental metrics exported from model training runs.
* **`README.md`**: Project overview and documentation.

---

## 🛠️ Requirements & Dependencies

To replicate the experiments or process evaluation logs, install the required packages:

```bash
pip install pandas numpy scikit-learn torch torchvision
```

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Explore the performance metrics in `Task_01_Results.md` or review the Word document `Task 01 - Updated.docx`.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
