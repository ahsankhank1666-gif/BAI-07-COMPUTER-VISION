# Computer Vision Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

**Student Name:** Muhammad Ahsan Khan  
**Registration Number:** FA23-BAI-052  
**Department:** Computer Science (BS Artificial Intelligence)  
**Institution:** COMSATS University Islamabad, Wah Campus  

---

## 📌 Project Overview
This repository contains the full implementation and experimental analysis for **Lab 03: Edge Detection Techniques and Their Impact on Classification Performance**. The lab explores first-order, second-order, and multi-stage edge detection operators, evaluates their sensitivity to artificial noise, and analyzes how classical edge representations influence machine learning classification outcomes compared to raw and filtered input images.

---

## 📁 Repository Structure
```text
lab03-cv/
├── lab03_pipeline.py          # Main execution script (Tasks 1-6)
├── README.md                  # Project documentation
├── Lab03_Report.md            # Comprehensive lab report & answers
└── results/                   # Generated output artifacts
    ├── Task1_Edge_Comparison.png
    ├── Task2_Noise_Effects.png
    ├── Task3_Canny_Parameters.png
    ├── Task6_BarChart.png
    ├── Task6_CM_Raw.png
    ├── Task6_CM_Filtered.png
    └── Task6_CM_Edge.png

    ⚙️ How to Setup and Run
1.Clone the repository:
git clone <your-repository-url>
cd lab03-cv

2.Install required dependencies:
pip install opencv-python numpy matplotlib scikit-learn

3.Execute the automation pipeline:
python lab03_pipeline.py

4.Outputs: All generated figures, confusion matrices, and performance bar charts will automatically save to the results/ folder.

