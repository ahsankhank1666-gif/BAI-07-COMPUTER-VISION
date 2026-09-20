# Lab Task 02: Effect of Image Filtering on Skin-Lesion Classification

**Author:** Muhammad Ahsan Khan (FA23-BAI-052)  
**Program:** BS Artificial Intelligence  
**Institution:** COMSATS University Islamabad, Wah Campus  

## Overview
This repository contains the experimental code to investigate how spatial-domain image-processing filters (Average, Gaussian, Median, Sharpening, and Sobel) affect the performance of pretrained deep-learning models (ResNet18, DenseNet121, EfficientNetB0) on the HAM10000 skin-lesion dataset[cite: 14].

## Repository Contents
*   `Lab_02_Skin_Lesion_Filtering.ipynb`: The complete executable Jupyter/Colab notebook containing dataset preparation, filtering logic, model training, evaluation, and visualizations[cite: 14].
*   `Task 02.docx`: The final lab report containing the comparative metrics table and analytical answers[cite: 14].
*   `README.md`: Execution instructions and project structure[cite: 14].

## Prerequisites
To run the experiments, ensure your Python environment has the following installed:
*   PyTorch & Torchvision
*   OpenCV (`opencv-python`)
*   Pandas, NumPy, Matplotlib, Seaborn
*   Scikit-Learn
*   Kagglehub (`pip install kagglehub`)

## How to Run the Experiments
1. **Environment Setup:** Open `Lab_02_Skin_Lesion_Filtering.ipynb` in Google Colab or Kaggle Notebooks.
2. **Hardware Acceleration:** Ensure a GPU runtime is active (e.g., in Colab: *Runtime > Change runtime type > T4 GPU*).
3. **Dataset Acquisition:** Do not manually download the dataset. The code utilizes `kagglehub` to automatically fetch the HAM10000 dataset directly into the notebook's environment.
4. **Execution:** Run the cells sequentially from top to bottom. 
5. **Adjusting Compute Time:** Training three separate architectures across six filtering states is highly compute-intensive. For a full run, leave the configuration as is. For a rapid functionality test, locate the `EPOCHS` variable in the "Experiment Execution" section and reduce it to `1` before executing the loop.