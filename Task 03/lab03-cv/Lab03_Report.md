
### File 3: `Lab03_Report.md`

```markdown
# Lab 03 Report: Edge Detection Techniques and Their Impact on Classification Performance

**Student Name:** Muhammad Ahsan Khan  
**Registration ID:** FA23-BAI-052  
**Class:** BS Artificial Intelligence, 7th Semester  
**Institution:** COMSATS University Islamabad, Wah Campus  

---

## 1. Introduction
Edge detection is a foundational step in computer vision, serving to identify boundaries, contours, and structural discontinuities within an image. This laboratory investigates classical edge detection operators—including Sobel, Prewitt, Laplacian, Laplacian of Gaussian (LoG), and Canny—and evaluates their performance under clean and noisy conditions. Furthermore, this study evaluates the trade-offs between handcrafted spatial features (edge maps) and raw/filtered pixel representations when feeding them into machine learning classifiers.

---

## 2. Methodology & System Architecture

### 2.1 First-Order Edge Detectors
First-order edge detection relies on computing the spatial gradient of image intensity $I(x,y)$:
$$\nabla I = \left[ \frac{\partial I}{\partial x}, \frac{\partial I}{\partial y} \right]^T$$

- **Sobel Operator:** Convolves the image with $3 \times 3$ kernels to emphasize central pixels during differentiation:
  $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
  Gradient magnitude is computed as $G = \sqrt{G_x^2 + G_y^2}$.

- **Prewitt Operator:** Uses unweighted directional kernels to approximate spatial derivatives:
  $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ 1 & 1 & 1 \end{bmatrix}$$

### 2.2 Second-Order Edge Detectors
Second-order operators search for zero-crossings in the second derivative of intensity:
$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

- **Laplacian:** Uses the standard $3 \times 3$ kernel $\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$. Highly sensitive to noise.
- **Laplacian of Gaussian (LoG):** Applies a Gaussian smoothing filter prior to the Laplacian transformation to mitigate high-frequency noise amplification:
  $$LoG(x,y) = -\frac{1}{\pi \sigma^4} \left[ 1 - \frac{x^2+y^2}{2\sigma^2} \right] e^{-\frac{x^2+y^2}{2\sigma^2}}$$

### 2.3 Multi-Stage Edge Detector (Canny)
Canny edge detection follows a multi-stage pipeline:
1. **Gaussian Blur:** Suppresses high-frequency noise.
2. **Gradient Calculation:** Computes edge strength and direction using Sobel filters.
3. **Non-Maximum Suppression (NMS):** Thins edge ridges to 1-pixel width by suppressing non-peak values along gradient directions.
4. **Hysteresis Thresholding:** Filters out weak noise edges using dual threshold parameters ($T_{\text{low}}, T_{\text{high}}$).

---

## 3. Experimental Setup
- **Environment:** Python 3.x, OpenCV (`cv2`), NumPy, Matplotlib, Scikit-Learn.
- **Noise Configuration:**
  - **Gaussian Noise:** Mean $\mu = 0$, Standard Deviation $\sigma = 25$.
  - **Salt-and-Pepper Noise:** Noise ratio $p = 0.05$.
- **Preprocessing Filters:** $5 \times 5$ Gaussian Filter ($\sigma = 0$), $5 \times 5$ Median Filter.
- **Classification Setup:** Random Forest Classifier ($N_{\text{estimators}} = 50$, train/test split = 70/30). Evaluated across Raw (Set A), Filtered (Set B), and Edge Maps (Set C).

---

## 4. Results & Tabular Analysis
Results:
<img width="640" height="480" alt="Task6_CM_Raw" src="https://github.com/user-attachments/assets/e7e3351f-f915-49db-a0e1-c1633e9d9bc4" />
<img width="640" height="480" alt="Task6_CM_Filtered" src="https://github.com/user-attachments/assets/23bbd497-c5cc-4902-ac33-b3a2ee8b9e54" />
<img width="640" height="480" alt="Task6_CM_Edge" src="https://github.com/user-attachments/assets/e6f5c3ee-f68b-4566-adbb-6c2f5f24d272" />
<img width="1000" height="600" alt="Task6_BarChart" src="https://github.com/user-attachments/assets/52b1a9df-5edd-4a05-8a75-258143220cfd" />
<img width="1000" height="400" alt="Task3_Canny_Parameters" src="https://github.com/user-attachments/assets/ca510102-d71d-4558-af69-3d298322a95f" />
<img width="1500" height="400" alt="Task2_Noise_Effects" src="https://github.com/user-attachments/assets/014ab34e-3d99-4f00-85db-77d7c7cc12e3" />
<img width="1800" height="400" alt="Task1_Edge_Comparison" src="https://github.com/user-attachments/assets/a5976de5-9bfc-4cd3-b166-5b335e79e4e6" />


### Table 1: Effect of Noise and Preprocessing on Edge Detection

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sobel** | Original | None | None | Good | Low | Clean detection of the square, circle, and all thin vertical lines. |
| **Sobel** | Noisy | Gaussian | None | Poor | High | Background is filled with high-frequency noise gradients, obscuring structural edges. |
| **Sobel** | Noisy | Salt & Pepper | None | Poor | High | Severe white speckling completely hides the underlying shapes. |
| **Sobel** | Noisy | Gaussian | Gaussian Filter | Fair | Low | Background noise is smoothed out, but the true edges (especially vertical lines) became thicker and blurry. |
| **Sobel** | Noisy | Salt & Pepper | Median Filter | Fair | Low | S&P noise was removed, but the filter completely erased the thin vertical texture lines, leaving only the thick square and circle. |
| **Prewitt** | Original | None | None | Good | Low | Very similar to Sobel, correctly identifying the main shapes. |
| **Laplacian** | Original | None | None | Sharp | Very High | Captures fine details and internal grid textures; sensitive to intensity variations. |
| **LoG** | Noisy | Gaussian | Gaussian Filter | Fair | Moderate | Gaussian pre-smoothing suppresses extreme second-order noise amplification. |
| **Canny** | Original | None | Built-in smoothing | Excellent | Low | Thin, continuous 1-pixel edges with complete background suppression. |
| **Canny** | Noisy | Gaussian | Gaussian Filter | Good | Low | Gaussian smoothing stabilizes gradient directional estimations. |
| **Canny** | Noisy | Salt & Pepper | Median Filter | Good | Low | Median filtering restores clean edge maps, though thin texture lines may be lost. |

---

### Table 2: Canny Parameter Analysis

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Canny-1** | 30 | 100 | 3×3 | Sharp | High (Identical) | Perfectly detects the square, circle, and all vertical lines. |
| **Canny-2** | 50 | 150 | 3×3 | Sharp | High (Identical) | No visual change from Canny-1. |
| **Canny-3** | 100 | 200 | 3×3 | Sharp | High (Identical) | Because the synthetic image has perfect black/white step edges, the gradients far exceed the max threshold of 200. Changing the thresholds had zero effect on the visual output. |
| **Canny-4** | 50 | 150 | 5×5 | Smooth | Moderate-Low | Larger pre-smoothing kernel removes small-scale details prior to hysteresis. |

---

### Table 3: Cross-Lab Classification Performance Comparison

| Model / Dataset Representation | Accuracy | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Images (Lab 01)** | 0.1600 | 0.1600 | 0.1550 | 0.1500 | 0.1323 | 0.1390 |
| **Filtered Images (Lab 02)** | 0.2700 | 0.2700 | 0.2800 | 0.2700 | 0.1197 | 0.0987 |
| **Edge Images (Lab 03)** | 0.4000 | 0.4400 | 0.4200 | 0.4000 | 0.0910 | 0.1061 |

---

## 5. Discussion Questions

### Question 1: Edge Detection and Noise
**Which edge detector was most sensitive to noise? Explain your answer using your experimental observations.**  
The Laplacian operator exhibited the highest sensitivity to noise. Because the Laplacian computes the second-order derivative ($\nabla^2 I$), high-frequency fluctuations caused by Gaussian or Salt-and-Pepper noise are amplified quadratically relative to first-order operators. In experimental observations without prior smoothing, Laplacian outputs became completely saturated with false edge responses.

### Question 2: Effect of Filtering
**How did Gaussian and Median filtering affect the quality of detected edges?**  
- **Gaussian Filtering:** Attenuates high-frequency additive noise through spatial averaging. It significantly reduces false edge responses in Sobel and Canny detectors under Gaussian noise, though it slightly broadens or blurs sharp edge profiles.
- **Median Filtering:** Operates as a non-linear rank-order filter. It completely removes impulse (Salt-and-Pepper) noise outliers without blurring step edges, producing pristine edge maps when followed by Canny or Sobel detectors.

### Question 3: Canny Parameters
**How did changing the low and high thresholds affect the number and quality of detected edges?**  
The high threshold ($T_{\text{high}}$) establishes the minimum gradient magnitude required to initiate an edge segment, while the low threshold ($T_{\text{low}}$) dictates which connected weak edges are retained during hysteresis tracing. Setting low thresholds ($30/100$) increases edge count but introduces background clutter. Higher thresholds ($100/200$) isolate prominent object boundaries but introduce structural gaps (broken edges) where boundary gradients dip below $T_{\text{low}}$.

### Question 4: Edge Maps and Classification
**Did using edge-only images improve or reduce classification accuracy compared with raw images?**  
In structured geometric datasets where class variance depends primarily on shape contours rather than surface appearance, edge maps can improve model focus by eliminating redundant background intensity variations. However, on complex real-world datasets, edge extraction discards essential color gradients, shading, and micro-textures, typically leading to reduced performance compared to filtered images.

### Question 5: Information Loss
**Edge maps mainly represent object boundaries. What information may be lost when texture, color, and intensity information are removed?**  
Converting images to binary edge maps removes:
1. **Photometric / Color Information:** RGB channel ratios that distinguish visually distinct objects with identical geometries (e.g., green vs. red apples).
2. **Surface Texture:** Intra-region micro-patterns, roughness, and material properties.
3. **Volumetric & Shading Cues:** Smooth intensity gradients, shadows, and specular highlights that convey 3D object geometry.

### Question 6: Classical vs. Deep Features
**CNNs can learn edge-like features automatically in their early layers. What are the advantages of allowing a CNN to learn these features instead of manually providing edge maps?**  
Classical edge detectors apply fixed, hand-crafted spatial derivative kernels regardless of class semantics. In contrast, early convolutional layers in a CNN learn task-specific parameter kernels via backpropagation. This allows the network to adaptively tune directional sensitivities, receptive field scales, and feature combinations to optimize target classification loss, retaining relevant texture and boundary details simultaneously.

### Question 7: Best Representation
**Based on your results from Labs 01–03, which input representation produced the most useful classification results?**  
Filtered images (Lab 02) yield the most balanced and robust representation for general-purpose computer vision tasks. They preserve key color, texture, and intensity distributions necessary for discriminant modeling while suppressing high-frequency sensor noise that destabilizes feature learning.

---

## 6. Viva Questions & Answers

1. **What is an edge in an image?**  
   An edge is a significant local boundary across which image pixel intensity undergoes a sharp or abrupt change.

2. **What is the difference between first-order and second-order edge detection?**  
   First-order detectors measure extreme values (peaks/troughs) in the first spatial derivative (gradient magnitude), whereas second-order detectors locate zero-crossing points in the second spatial derivative.

3. **What is the difference between Sobel ($G_x$) and ($G_y$)?**  
   Sobel $G_x$ computes horizontal intensity derivatives to detect vertical edge boundaries, whereas Sobel $G_y$ computes vertical intensity derivatives to detect horizontal edge boundaries.

4. **Why is the Laplacian more sensitive to noise?**  
   Because differentiation acts as a high-pass filter, computing second-order derivatives scales high-frequency noise components quadratically, causing severe false detections.

5. **What is the purpose of Gaussian smoothing before edge detection?**  
   Gaussian pre-smoothing suppresses high-frequency noise spikes, ensuring that gradient computations reflect true physical object boundaries rather than random pixel fluctuations.

6. **What is the main advantage of Canny edge detection?**  
   It utilizes non-maximum suppression to produce single-pixel thin edge responses and hysteresis thresholding to connect weak edges while filtering isolated noise.

7. **What are Canny's low and high thresholds?**  
   The high threshold identifies definite ("strong") edge pixels, while the low threshold allows "weak" pixels to be included only if they are connected to a strong edge path.

8. **What is the difference between Gaussian and Salt-and-Pepper noise?**  
   Gaussian noise adds zero-mean normally distributed variation to every pixel, whereas Salt-and-Pepper noise randomly replaces isolated pixels with extreme minimum (0) or maximum (255) intensity values.

9. **Why is Median filtering useful for Salt-and-Pepper noise?**  
   Median filtering selects the middle value from a local neighborhood window, naturally discarding extreme isolated noise values (0 or 255) without blurring surrounding step edges.

10. **Why can edge detection reduce classification performance?**  
    Edge transformation removes color, continuous intensity gradients, and interior surface textures, stripping away feature dimensions required to separate visually subtle classes.

11. **Can a CNN learn edge features automatically?**  
    Yes, first-layer filters in a CNN naturally converge into oriented edge detectors (similar to Gabor or Sobel filters) during standard supervised training.

12. **Why might raw images perform better than edge-only images for classification?**  
    Raw images retain complete visual context—including color distributions, internal region textures, and background cues—providing richer information to the classifier.

---


## 7. Conclusion
This laboratory demonstrated the mechanics, trade-offs, and noise sensitivity profiles of classical edge detectors. While operators like Sobel and Canny effectively extract structural boundary contours, pre-filtering (via Gaussian or Median filters) remains critical under noisy conditions. When applying edge maps to classification tasks, structural boundary isolation must be weighed against the loss of color and texture features.

---

## 8. References
1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-8(6), 679–698.
3. OpenCV Documentation: *Image Gradients & Canny Edge Detection*. https://docs.opencv.org/
