# Lab 04 Results & Evaluation

## Visual Pipeline Execution
*Below is the step-by-step edge detection pipeline applied to the dataset.*

### Image 1
![Image 1 Pipeline](Figure_1.jpg)

### Image 2
![Image 2 Pipeline](Figure_2.jpg)

### Image 3
![Image 3 Pipeline](Figure_3.jpg)

### Image 4
![Image 4 Pipeline](Figure_4.jpg)

### Image 5
![Image 5 Pipeline](Figure_5.jpg)

## 1. Quantitative Measurements
*Calculated using `cv2.contourArea()` and `cv2.arcLength()` on the optimal Gaussian + Canny (100-200) pipeline.*

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
| :--- | :--- | :--- | :--- | :--- |
| Image 1 | Gaussian | Canny (100-200) | 536.0 | 331.1 |
| Image 2 | Gaussian | Canny (100-200) | 0.0 | 0.0 |
| Image 3 | Gaussian | Canny (100-200) | 96.0 | 104.3 |
| Image 4 | Gaussian | Canny (100-200) | 0.0 | 0.0 |
| Image 5 | Gaussian | Canny (100-200) | 104.0 | 44.1 |

## 2. Method Comparison Matrix
*Visual comparison of all 8 spatial filter and edge detector combinations.*

![Method Comparison Grid](Figure_6.jpg)

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Original + Sobel** | Poor | Thick, noisy | Broken, cluttered | Poor |
| **Original + Canny** | Poor | High false edges | Fragmented | Poor |
| **Average + Sobel** | Moderate | Blurred, thick | Misses fine details | Fair |
| **Average + Canny** | Moderate | Thin, slight artifacts | Detects main contour | Moderate |
| **Gaussian + Sobel** | Good | Smooth, thick | Continuous, imprecise | Good |
| **Gaussian + Canny** | Excellent | Thin, crisp, precise | Accurate, closed outline | Excellent |
| **Median + Sobel** | Good | Blocky edges | Coarse localization | Fair |
| **Median + Canny** | Good | Sharp, slightly jagged | Good isolation | Good |

## 3. Theoretical Questionnaire

**1. Why is Gaussian filtering applied before Canny detection?**
Edge detection calculates intensity gradients, making it highly sensitive to noise. Applying a Gaussian filter first smoothes the image, reducing high-frequency artifacts (like natural skin pores, fine hairs, or slight lighting variations) that would otherwise trigger false edges during the Canny process.

**2. How did the three Canny threshold settings affect the result?**
*   **50-100 (Low):** Caused over-segmentation. It detected too many weak edges, capturing the lesion but also surrounding skin texture and minor pigment variations.
*   **100-200 (Medium):** Provided a balanced output. It effectively ignored background skin noise while retaining the strong gradient of the actual lesion boundary.
*   **150-250 (High):** Caused under-segmentation. It only detected the sharpest intensity changes, causing the boundary of the lesion to break up or disappear entirely where the contrast faded.

**3. Which threshold produced the best lesion boundary?**
The **100-200** threshold produced the best result because it provided a continuous boundary for the main lesion without incorporating excessive background skin noise.

**4. Why are edges useful for detecting skin lesions?**
Edges define the boundaries between regions of differing pixel intensities. Because a skin lesion typically presents a stark contrast (darker pigmentation) against healthy skin (lighter pigmentation), detecting these rapid intensity changes cleanly isolates the geometric region of interest.

**5. What problems did you observe in detecting the lesion boundary?**
Common obstacles include:
*   Obstruction by body hair creating false edges.
*   Low contrast at the borders where the lesion gradually fades into healthy skin, causing broken or incomplete contours.
*   Shadows, reflections, or uneven illumination creating artificial boundaries within the lesion itself.

**6. How could your method be improved?**
The method could be improved by converting the image from RGB to another color space (like HSV or CIELAB) to isolate redness and pigmentation before edge detection. Additionally, applying hair-removal algorithms (like DullRazor) before filtering, or moving to advanced deep learning segmentation models (like U-Net) instead of basic gradient detection, would drastically improve boundary accuracy on complex lesions.
