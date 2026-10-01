# Skin Lesion Boundary Detection
**Computer Vision Lab Assignment 04**

## Overview
This repository contains a computer vision pipeline developed to detect the boundaries of skin lesions using spatial filtering and Canny edge detection. The project evaluates multiple preprocessing filters (Gaussian, Average, Median) alongside edge detectors (Canny, Sobel) to determine the optimal configuration for noise handling, edge quality, and boundary continuity.

## Dataset
Images are automatically sourced from the public **HAM10000** (Human Against Machine) dataset using `kagglehub`. The script dynamically selects the first 5 images for processing and evaluation.

## Prerequisites
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Kagglehub

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/cv-lab04-skin-lesion.git](https://github.com/YourUsername/cv-lab04-skin-lesion.git)
   cd cv-lab04-skin-lesion

  1. Install dependencies:
  pip install -r requirements.txt

  2. Execute the pipeline:
  python main.py

  Methodology
1 Grayscale Conversion: Simplifies the image matrix.

2 Gaussian Filtering (5x5): Suppresses high-frequency skin texture and hair noise.

3 Canny Edge Detection (100-200): Isolates the stark gradient changes at the lesion's perimeter.

4 Morphological Closing: Closes small gaps in the detected edge map.

5 Contour Extraction: Overlays the continuous boundary and calculates area and perimeter. 

### Step 4: Run the Code and Capture Results
1. In your terminal, run the script:
   ```bash
   python main.py

   
1. The script will download the dataset and pop up Matplotlib windows showing the image processing pipelines.

2. Save your visual results: In the Matplotlib window, click the "Save" (floppy disk) icon at the bottom. Save the 5 individual image pipeline plots as result_img1.png, result_img2.png, etc., and save the final 8-method comparison grid as comparison_grid.png. Place these in your project folder.

3. Close the plot windows to allow the script to finish running. The terminal will then print the two required tables.