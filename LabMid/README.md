# Computer Vision Lab Midterm: HOG vs SIFT on Blood-Cell Images

**Name:** Muhammad Ahsan Khan  
**Registration:** FA23-BAI-052  
**Dataset:** BloodMNIST (Grayscale, 128x128)  

## Personal Settings (Seed: 52)
* **HOG Cell Size:** 8 x 8 (Registration last digit '2' is even)
* **SIFT Grid Size:** 4 x 4 grid / 16 keypoints (Last digit '2' is between 0-4)

## Q4. Robustness Evaluation
The models were tested under a 20-degree rotation and Gaussian noise ($\sigma = 0.05$).

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 0.6917 | 0.6583 | 0.5083 |
| SIFT + SVM | 0.7083 | 0.5000 | 0.6917 |

**Conclusion:** SIFT was highly vulnerable to rotation (dropping to 0.5000) because forcing it onto a fixed grid without its normal orientation assignment breaks its rotational invariance. However, SIFT resisted noise exceptionally well because it applies a Gaussian blur to patches before computing gradients, whereas HOG (which dropped to 0.5083) uses unsmoothed derivative masks, making it sensitive to pixel noise.

## Q5. Short Answers
* **(a) HOG Block Normalization:** Normalizes histograms over overlapping blocks to make features invariant to changes in local illumination and contrast.
* **(b) Test Set for C:** Tuning hyperparameters on the test set causes data leakage. The test set must remain unseen to provide an unbiased evaluation of real-world performance.
* **(c) Class Confusion:** The models most frequently confused Eosinophils (1) and Neutrophils (6). Converting the images to grayscale removes the primary distinguishing biological feature (pink vs. purple granules), forcing the SVM to rely on nearly identical morphological textures.

## AI Log
| Prompt (One line) | What was changed | Validation |
| :--- | :--- | :--- |
| "Explain how medmnist.imgs and dataset.labels work." | Wrote the `process_split` function. | Printed `X_train.shape` to verify `(400, 128, 128)`. |
| "Calculate HOG and fixed-grid SIFT feature lengths." | Implemented `extract_hog` with 8x8 cells and 4x4 SIFT grid. | Verified `X_train_hog.shape[1]` printed `8100` and `X_train_sift.shape[1]` printed `2048`. |