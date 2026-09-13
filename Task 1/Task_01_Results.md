# Deep Learning Transfer Learning & Classifier Benchmark Report (Task 01)

This repository contains experimental benchmarking results and evaluation reports for transfer learning models, classical classifiers, and computational efficiency analyses on image classification tasks.

## Table of Contents
- [Overview](#overview)
- [Results & Performance Tables](#results--performance-tables)
  - [Table 1: Comparison of Transfer Learning Models](#table-1-comparison-of-transfer-learning-models)
  - [Table 2: Comparison of Different Classifiers](#table-2-comparison-of-different-classifiers)
  - [Table 3: Computational Efficiency Comparison](#table-3-computational-efficiency-comparison)
- [Repository Structure](#repository-structure)
- [Usage](#usage)

---

## Overview

This project evaluates the performance of state-of-the-art Convolutional Neural Networks (CNNs) used as feature extractors and transfer learning models, combined with various classical machine learning classifiers. Metrics reported include Accuracy, Precision, Recall, F1-Score, Area Under the ROC Curve (AUC), Parameter Count, Model Size, FLOPs, and Inference Time.

---

## Results & Performance Tables

### Table 1. Comparison of Transfer Learning Models
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AlexNet | 58.75 | 63.38 | 58.75 | 55.22 | 88.30 |
| VGG16 | 61.25 | 69.52 | 61.25 | 54.31 | 88.16 |
| VGG19 | 60.00 | 64.22 | 60.00 | 58.62 | 88.69 |
| ResNet18 | 61.25 | 63.04 | 61.25 | 54.84 | 90.72 |
| ResNet50 | 57.50 | 58.89 | 57.50 | 54.58 | 87.29 |
| ResNet101 | 58.75 | 54.49 | 58.75 | 51.91 | 92.42 |
| DenseNet121 | 61.25 | 58.23 | 61.25 | 55.70 | 91.41 |
| EfficientNet-B0 | 62.50 | 65.65 | 62.50 | 58.08 | 90.84 |

### Table 2. Comparison of Different Classifiers
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
| AlexNet | 57.02 | 217.54 | 0.71 | 2.10 | 58.75 |
| VGG16 | 134.28 | 512.25 | 15.47 | 10.68 | 61.25 |
| VGG19 | 139.59 | 532.51 | 19.63 | 12.89 | 60.00 |
| ResNet18 | 11.18 | 42.72 | 1.82 | 2.37 | 61.25 |
| ResNet50 | 23.52 | 90.02 | 4.13 | 5.62 | 57.50 |
| DenseNet121 | 6.96 | 27.13 | 2.90 | 19.29 | 61.25 |
| EfficientNet-B0 | 4.01 | 15.60 | 0.41 | 10.29 | 62.50 |

---

## Repository Structure

```text
├── Task 01 - Updated.docx     # Updated Word document containing populated results for all 3 tables
├── Task_01_Results.md         # Markdown version of Task 01 report for GitHub
├── table1_transfer_learning_results.csv # Raw CSV data file
└── README.md                  # Project overview and instructions