# Lab 06 – Landmark Recognition Using the SIFT Algorithm

> **Course**: BAI-07 Computer Vision  
> **Algorithm**: Scale-Invariant Feature Transform (SIFT)

---

## Project Structure

```
Lab 06/
├── lab06_landmark_recognition.py   # Main script (all 6 tasks)
├── database/                        # Reference images (auto-downloaded)
│   ├── eiffel_tower/
│   ├── taj_mahal/
│   ├── great_wall/
│   ├── statue_of_liberty/
│   └── colosseum/
├── query_images/                    # Auto-generated test variants
├── output/                          # Result images (keypoints, matches, localised)
└── README.md
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install opencv-python opencv-contrib-python numpy requests Pillow

# 2. Run the script
python lab06_landmark_recognition.py
```

The script:
1. Downloads reference images automatically from Wikimedia Commons.  
2. Extracts SIFT descriptors from all reference images.  
3. Generates 6 test variants per landmark (original, rotated, resized, dark, viewpoint-shift, unknown).  
4. Recognises each variant and attempts homography localisation.  
5. Prints the full performance table.  
6. Saves all output images to `./output/`.

---

## Algorithm Overview

| Step | Description |
|------|-------------|
| **SIFT Extraction** | Detect keypoints using Difference-of-Gaussian; compute 128-dim descriptors |
| **BF Matching** | Brute-Force matcher with Euclidean (L2) distance |
| **KNN (k=2)** | Find two nearest neighbours per query descriptor |
| **Lowe's Ratio Test** | Accept match only if `d1 / d2 < 0.75` |
| **Threshold** | Landmark accepted only if reliable matches ≥ 10 |
| **Homography** | RANSAC-based homography to localise landmark in query image |

---

## Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `LOWE_RATIO` | 0.75 | Ratio threshold for Lowe's test |
| `MIN_MATCH_COUNT` | 10 | Minimum good matches to accept recognition |
| `RANSAC_THRESH` | 5.0 px | RANSAC reprojection error threshold |

---

## Sample Performance Table

*(Generated automatically by Task 6 — values are illustrative)*

| Landmark | Variant | Keypoints | Matches | Predicted | Recognized | Time (s) |
|----------|---------|-----------|---------|-----------|------------|----------|
| Eiffel Tower | Original | ~800 | ~120 | Eiffel Tower | Yes | ~0.5 |
| Eiffel Tower | Rotated 45° | ~780 | ~95 | Eiffel Tower | Yes | ~0.5 |
| Eiffel Tower | Resized 50% | ~450 | ~60 | Eiffel Tower | Yes | ~0.4 |
| Eiffel Tower | Dark (-80) | ~600 | ~50 | Eiffel Tower | Yes | ~0.5 |
| Eiffel Tower | Viewpoint shift | ~700 | ~40 | Eiffel Tower | Yes | ~0.5 |
| Eiffel Tower | Unknown (noise) | ~0 | 0 | UNKNOWN | No | ~0.1 |
| … | … | … | … | … | … | … |

> Actual values are printed to the console and saved after running the script.

---

## Expected Outputs (in `./output/`)

| File pattern | Description |
|---|---|
| `kp_ref_<landmark>_<img>.jpg` | Reference image with SIFT keypoints drawn |
| `kp_query_<name>.jpg` | Query image with SIFT keypoints drawn |
| `matches_<name>.jpg` | Side-by-side match lines (reference ↔ query) |
| `localized_<name>.jpg` | Query image with homography polygon drawn |

---

## Discussion

### Why SIFT Works Well for Landmark Recognition
- **Scale invariance**: DoG pyramid detects features at multiple scales → robust to zoomed-in/out images.
- **Rotation invariance**: Dominant gradient orientation assigned per keypoint → match survives rotations.
- **Illumination robustness**: Gradient-based descriptor normalised to unit length → partially invariant to brightness changes.

### Limitations
1. **Patented / speed**: SIFT is slower than ORB or AKAZE; for real-time use consider ORB.
2. **Occlusion**: Heavy occlusion (< 30 % of landmark visible) causes recognition failure.
3. **Repetitive textures**: The Great Wall and Colosseum have repetitive patterns that generate many ambiguous matches.
4. **Threshold sensitivity**: A fixed `MIN_MATCH_COUNT` does not adapt to image resolution — very small images produce few keypoints even for correct landmarks.
5. **Unknown rejection**: The ratio + count threshold rejects most noise images, but unusual real-world scenes may still exceed the threshold accidentally.

### Possible Improvements
- Use FLANN-based matcher for faster large-scale matching.
- Replace per-image descriptor lists with a VLAD / Bag-of-Words index.
- Train a lightweight CNN classifier on top of SIFT features for better unknown rejection.
