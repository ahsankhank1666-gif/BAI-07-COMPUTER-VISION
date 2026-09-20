# 🔬 Skin-Lesion Classification: Effect of Image Filtering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository investigates how classical spatial-domain image-processing filters affect the performance of pretrained deep-learning models for medical image classification. Using the **HAM10000 skin-lesion dataset**, this project evaluates three state-of-the-art CNN architectures against multiple filter states (smoothing, sharpening, and edge detection) to understand the relationship between manual feature engineering and deep-learning feature extraction.

---

## 📊 Experimental Results Summary

The table below outlines the performance metrics of each model across different filtering states. 

*Note: The dataset is highly imbalanced; therefore, Macro-F1 and Balanced Accuracy serve as the primary indicators of true model robustness.*

| Model | Filter Applied | Accuracy (%) | Precision | Recall | Macro-F1 | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet18** | 🟢 None (Baseline) | 65.20 | 0.5052 | 0.4347 | 0.4191 | 0.8837 |
| **ResNet18** | 🟡 Average | 69.50 | 0.4903 | 0.4169 | 0.4163 | 0.8673 |
| **ResNet18** | 🟡 Gaussian | 75.49 | 0.5926 | 0.4316 | 0.4294 | 0.9110 |
| **ResNet18** | 🟡 Median | 75.89 | 0.5954 | 0.3855 | 0.4372 | 0.9313 |
| **ResNet18** | 🔵 Sharpening | 78.83 | 0.6006 | 0.4279 | 0.4834 | 0.9318 |
| **ResNet18** | 🔴 Sobel (Edge) | 67.30 | 0.3734 | 0.1508 | 0.1312 | 0.6999 |
| **DenseNet121**| 🟢 None (Baseline) | 78.43 | 0.6342 | 0.5095 | 0.5291 | 0.9303 |
| **DenseNet121**| 🟡 Average | 76.69 | 0.7053 | 0.4848 | 0.5031 | 0.9337 |
| **DenseNet121**| 🟡 Gaussian | 76.73 | 0.6807 | 0.5014 | 0.4912 | 0.9407 |
| **DenseNet121**| 🟡 Median | 77.88 | 0.5258 | 0.5256 | 0.5080 | 0.9397 |
| **DenseNet121**| 🔵 Sharpening | 75.64 | 0.5694 | 0.5605 | 0.5067 | 0.9336 |
| **DenseNet121**| 🔴 Sobel (Edge) | 59.51 | 0.2431 | 0.2613 | 0.2440 | 0.7257 |
| **EfficientNetB0**| 🟢 None (Baseline) | 83.52 | 0.6859 | 0.7315 | **0.6945** | **0.9696** |
| **EfficientNetB0**| 🟡 Average | 82.23 | 0.7233 | 0.6718 | 0.6869 | 0.9646 |
| **EfficientNetB0**| 🟡 Gaussian | **83.92** | **0.7503** | 0.6991 | 0.7148 | 0.9639 |
| **EfficientNetB0**| 🟡 Median | 81.13 | 0.7223 | 0.6274 | 0.6486 | 0.9595 |
| **EfficientNetB0**| 🔵 Sharpening | 82.28 | 0.7219 | 0.6840 | 0.6772 | 0.9631 |
| **EfficientNetB0**| 🔴 Sobel (Edge) | 75.39 | 0.6746 | 0.4126 | 0.4574 | 0.9085 |

> **Key Insight:** Applying classical spatial filters generally **degrades** deep learning performance on dermatological datasets. Smoothing obscures essential textural features (like pigment networks), and edge detection (Sobel) destroys critical color data. Deep models perform best on raw, unfiltered RGB inputs where they can learn their own optimal feature hierarchies.

---

## 📂 Repository Contents

* **`Lab_02_Skin_Lesion_Filtering.ipynb`**: The primary executable Google Colab / Kaggle Notebook containing dataset preparation, OpenCV filtering logic, training loops, and evaluation engines.
* **`Task 02.docx`**: The comprehensive lab report containing full analytical answers, comparative matrices, training graphs, and visual filter examples.
* **`README.md`**: Project overview and setup documentation.

---

## 🛠️ Requirements & Dependencies

To replicate the experiments locally or in a cloud notebook environment, install the required packages:

```bash
pip install torch torchvision opencv-python pandas numpy matplotlib seaborn scikit-learn kagglehub