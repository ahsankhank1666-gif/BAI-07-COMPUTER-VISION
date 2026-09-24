# Computer Vision Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

**Student Name:** Muhammad Ahsan Khan  
**Registration Number:** FA23-BAI-052  
**Department:** Computer Science (BS Artificial Intelligence)  
**Institution:** COMSATS University Islamabad, Wah Campus  

---

## 📌 Project Overview
This repository contains the full implementation and experimental analysis for **Lab 03: Edge Detection Techniques and Their Impact on Classification Performance**. The lab explores first-order, second-order, and multi-stage edge detection operators, evaluates their sensitivity to artificial noise, and analyzes how classical edge representations influence machine learning classification outcomes compared to raw and filtered input images.
---


Results:
<img width="640" height="480" alt="Task6_CM_Raw" src="https://github.com/user-attachments/assets/51c5ff6d-d3d0-49da-83b8-2816e14b9fdc" />
<img width="640" height="480" alt="Task6_CM_Filtered" src="https://github.com/user-attachments/assets/7cbf0995-bf27-4244-9ec0-704c77da7314" />
<img width="640" height="480" alt="Task6_CM_Edge" src="https://github.com/user-attachments/assets/60a10dc4-deb1-473d-8765-dfd3024ae90e" />
<img width="1000" height="600" alt="Task6_BarChart" src="https://github.com/user-attachments/assets/3a2d78ae-693b-4e9f-b024-1cf03b38d5e3" />
<img width="1000" height="400" alt="Task3_Canny_Parameters" src="https://github.com/user-attachments/assets/517fb8c2-9a14-4702-b318-07bd6a8c76ec" />
<img width="1500" height="400" alt="Task2_Noise_Effects" src="https://github.com/user-attachments/assets/d126f03e-7773-4762-aeb3-c54f6914c872" />
<img width="1800" height="400" alt="Task1_Edge_Comparison" src="https://github.com/user-attachments/assets/f86055ee-4c12-4506-8f2d-3c3c587a9562" />


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








