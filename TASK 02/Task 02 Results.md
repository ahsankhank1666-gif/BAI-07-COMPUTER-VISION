# Skin-Lesion Classification: Effect of Image Filtering (Task 02 Results)

This repository contains experimental benchmarking results investigating how different spatial-domain image-processing filters affect the performance of pretrained deep-learning models for skin-lesion classification[cite: 15].

## Table of Contents

* [Overview](https://www.google.com/search?q=%2523overview&utm_source=gemini)
* [Results & Performance Tables](https://www.google.com/search?q=%2523results--performance-tables&utm_source=gemini)
* [Comparison of Image Filtering on Pretrained Models](https://www.google.com/search?q=%2523comparison-of-image-filtering-on-pretrained-models&utm_source=gemini)


* [Analytical Insights](https://www.google.com/search?q=%2523analytical-insights&utm_source=gemini)
* [Repository Structure](https://www.google.com/search?q=%2523repository-structure&utm_source=gemini)
* [Usage](https://www.google.com/search?q=%2523usage&utm_source=gemini)

---

## Overview

This project evaluates the performance of three state-of-the-art Convolutional Neural Networks (ResNet18, DenseNet121, and EfficientNetB0) against multiple spatial filter states (Average, Gaussian, Median, Sharpening, and Sobel)[cite: 15]. The experiments utilize the complete HAM10000 skin-lesion dataset, mapped to 7 distinct classes (Melanocytic nevi, Melanoma, Benign keratosis-like lesions, Basal cell carcinoma, Actinic keratoses, Vascular lesions, and Dermatofibroma)[cite: 15].

---

## Results & Performance Tables

### Comparison of Image Filtering on Pretrained Models

| Model | Filter Applied | Accuracy | Precision | Recall | Macro-F1 | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| **ResNet18** | 🟢 No Filter | 0.6520 | 0.5052 | 0.4347 | 0.4191 | 0.8837[cite: 15] |
| **ResNet18** | 🟡 Average | 0.6950 | 0.4903 | 0.4169 | 0.4163 | 0.8673[cite: 15] |
| **ResNet18** | 🟡 Gaussian | 0.7549 | 0.5926 | 0.4316 | 0.4294 | 0.9110[cite: 15] |
| **ResNet18** | 🟡 Median | 0.7589 | 0.5954 | 0.3855 | 0.4372 | 0.9313[cite: 15] |
| **ResNet18** | 🔵 Sharpening | 0.7883 | 0.6006 | 0.4279 | 0.4834 | 0.9318[cite: 15] |
| **ResNet18** | 🔴 Sobel | 0.6730 | 0.3734 | 0.1508 | 0.1312 | 0.6999[cite: 15] |
| **DenseNet121** | 🟢 No Filter | 0.7843 | 0.6342 | 0.5095 | 0.5291 | 0.9303[cite: 15] |
| **DenseNet121** | 🟡 Average | 0.7669 | 0.7053 | 0.4848 | 0.5031 | 0.9337[cite: 15] |
| **DenseNet121** | 🟡 Gaussian | 0.7673 | 0.6807 | 0.5014 | 0.4912 | 0.9407[cite: 15] |
| **DenseNet121** | 🟡 Median | 0.7788 | 0.5258 | 0.5256 | 0.5080 | 0.9397[cite: 15] |
| **DenseNet121** | 🔵 Sharpening | 0.7564 | 0.5694 | 0.5605 | 0.5067 | 0.9336[cite: 15] |
| **DenseNet121** | 🔴 Sobel | 0.5951 | 0.2431 | 0.2613 | 0.2440 | 0.7257[cite: 15] |
| **EfficientNetB0** | 🟢 No Filter | 0.8352 | 0.6859 | 0.7315 | 0.6945 | 0.9696[cite: 15] |
| **EfficientNetB0** | 🟡 Average | 0.8223 | 0.7233 | 0.6718 | 0.6869 | 0.9646[cite: 15] |
| **EfficientNetB0** | 🟡 Gaussian | 0.8392 | 0.7503 | 0.6991 | 0.7148 | 0.9639[cite: 15] |
| **EfficientNetB0** | 🟡 Median | 0.8113 | 0.7223 | 0.6274 | 0.6486 | 0.9595[cite: 15] |
| **EfficientNetB0** | 🔵 Sharpening | 0.8228 | 0.7219 | 0.6840 | 0.6772 | 0.9631[cite: 15] |
| **EfficientNetB0** | 🔴 Sobel | 0.7539 | 0.6746 | 0.4126 | 0.4574 | 0.9085[cite: 15] |

---

## Analytical Insights


<img width="1280" height="576" alt="10" src="https://github.com/user-attachments/assets/fd00628c-8a78-4453-b71d-78150f08d8f9" />
<img width="1280" height="576" alt="9" src="https://github.com/user-attachments/assets/9b62a32d-0a44-40c8-95d5-0730d088ad92" />
<img width="1280" height="576" alt="8" src="https://github.com/user-attachments/assets/f30bfadf-0210-4bd8-9d6d-49279c0bd1db" />
<img width="1280" height="576" alt="2" src="https://github.com/user-attachments/assets/27ad80ac-696f-4f9b-ab58-c34bf0efd424" />
<img width="1280" height="576" alt="1" src="https://github.com/user-attachments/assets/c8b58e15-878f-46cc-8477-81cd0e649390" />
<img width="1280" height="576" alt="11" src="https://github.com/user-attachments/assets/5f8ab7f0-237f-4663-baf0-0270e2497758" />


* **Top Performing Model:** EfficientNetB0 demonstrated the highest baseline capability, achieving 83.52% accuracy and a 0.6945 Macro-F1 score without any filters[cite: 15].
* **Filter Impact:** Aggressive filtering generally degrades model performance[cite: 15]. Smoothing filters caused slight drops in Macro-F1 across DenseNet121 and EfficientNetB0, though Gaussian filtering showed a rare exception by slightly improving EfficientNetB0's Macro-F1 to 0.7148[cite: 15]. Sobel edge detection severely crippled all three models[cite: 15].
* **Greatest Disruption:** The Sobel filter produced the most drastic negative change across all architectures[cite: 15]. DenseNet121's Macro-F1 dropped from 0.5291 to 0.2440, and EfficientNetB0's Macro-F1 fell from 0.6945 to 0.4574[cite: 15].
* **Minority Class Vulnerability:** Filtering decreases the macro-F1 score by obscuring the subtle textures required to identify rare lesion types, forcing models to default to predicting the dominant class ('Melanocytic nevi')[cite: 15]. Minority classes like Dermatofibroma, Actinic keratoses, and Vascular lesions are disproportionately affected[cite: 15].
* **Feature Destruction vs. Extraction:** Smoothing acts as a low-pass filter, destroying the primary mathematical variance convolutional layers rely on[cite: 15]. Sobel entirely removes color data[cite: 15]. Applying rigid classical filters before deep learning restricts the model's potential by depriving the network of the complete RGB spatial data needed to independently extract optimal features[cite: 15].

---

## Repository Structure

```text

├── Task_02_Results.md            # GitHub documentation
├── Lab_02_Skin_Lesion.ipynb      # The complete executable notebook for the experimental pipeline
└── README.md                     # Project overview and instructions

```

---

