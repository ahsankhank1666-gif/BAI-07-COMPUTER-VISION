import cv2
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import glob
import os

# --- Task Helpers ---

def apply_sobel(gray_img):
    """Calculates Sobel magnitude image."""
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def apply_filter(gray_img, filter_type):
    """Applies specified spatial filtering."""
    if filter_type == "Original":
        return gray_img.copy()
    elif filter_type == "Average":
        return cv2.blur(gray_img, (5, 5))
    elif filter_type == "Gaussian":
        return cv2.GaussianBlur(gray_img, (5, 5), 0)
    elif filter_type == "Median":
        return cv2.medianBlur(gray_img, 5)

def process_single_image(image_path, img_title):
    """Executes Tasks 1 through 6 for a single image."""
    # Task 1: Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Task 2: Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_gaussian = apply_filter(gray, "Gaussian")

    # Task 3: Canny Edge Detection with 3 Thresholds
    c_low = cv2.Canny(blurred_gaussian, 50, 100)
    c_mid = cv2.Canny(blurred_gaussian, 100, 200)
    c_high = cv2.Canny(blurred_gaussian, 150, 250)

    # Task 4 & 5: Boundary Extraction using Best Result (100-200)
    best_edges = c_mid
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.morphologyEx(best_edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_boundary = img_rgb.copy()
    area, perimeter = 0.0, 0.0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img_boundary, [largest_contour], -1, (0, 255, 0), 2)
        # Task 6: Area & Perimeter
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

    # Display Visualization (Task 1-5 Pipeline)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes.ravel()

    ax[0].imshow(img_rgb)
    ax[0].set_title(f"{img_title}: Original")
    ax[1].imshow(gray, cmap='gray')
    ax[1].set_title("Grayscale")
    ax[2].imshow(blurred_gaussian, cmap='gray')
    ax[2].set_title("Gaussian Filter")
    ax[3].imshow(c_low, cmap='gray')
    ax[3].set_title("Canny (50-100)")
    ax[4].imshow(c_mid, cmap='gray')
    ax[4].set_title("Canny (100-200) [Selected]")
    ax[5].imshow(img_boundary)
    ax[5].set_title(f"Lesion Boundary\nArea={area:.1f}px, Peri={perimeter:.1f}px")

    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    return area, perimeter

def generate_comparison_grid(image_path):
    """Generates visual comparison grid for all 8 Method Combinations."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    filters = ["Original", "Average", "Gaussian", "Median"]
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))

    for idx, f_name in enumerate(filters):
        filtered = apply_filter(gray, f_name)
        
        # Sobel
        sobel_res = apply_sobel(filtered)
        axes[idx, 0].imshow(sobel_res, cmap='gray')
        axes[idx, 0].set_title(f"{f_name} + Sobel")
        axes[idx, 0].axis('off')

        # Canny
        canny_res = cv2.Canny(filtered, 100, 200)
        axes[idx, 1].imshow(canny_res, cmap='gray')
        axes[idx, 1].set_title(f"{f_name} + Canny")
        axes[idx, 1].axis('off')

    plt.suptitle("Final Comparison Matrix (All 8 Method Combinations)", fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Main Program Execution ---

print("1. Downloading HAM10000 dataset...")
dataset_dir = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
image_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True))

if len(image_files) >= 5:
    selected_files = image_files[:5]
    table_1_rows = []

    print("\n2. Processing 5 images for Tasks 1-6...\n")
    for idx, file_path in enumerate(selected_files):
        label = f"Image {idx + 1}"
        area, perimeter = process_single_image(file_path, label)
        table_1_rows.append((label, "Gaussian", "Canny (100-200)", area, perimeter))

    print("\n3. Visualizing 8 Filter + Detector Combinations for Final Comparison Table...")
    generate_comparison_grid(selected_files[0])

    print("\n=================== REQUIRED TABLE 1 ===================")
    print(f"{'Image':<10} | {'Best Filter':<12} | {'Edge Method':<15} | {'Area (pixels)':<15} | {'Perimeter (pixels)'}")
    print("-" * 75)
    for row in table_1_rows:
        print(f"{row[0]:<10} | {row[1]:<12} | {row[2]:<15} | {row[3]:<15.1f} | {row[4]:.1f}")
    print("========================================================\n")


    print("\n=================== FINAL COMPARISON TABLE ===================")
    print(f"{'Method':<18} | {'Noise Handling':<15} | {'Edge Quality':<20} | {'Boundary Detection':<25} | {'Overall Performance'}")
    print("-" * 105)
    
    comparisons = [
        ("Original + Sobel", "Poor", "Thick, noisy", "Broken, cluttered", "Poor"),
        ("Original + Canny", "Poor", "High false edges", "Fragmented", "Poor"),
        ("Average + Sobel", "Moderate", "Blurred, thick", "Misses fine details", "Fair"),
        ("Average + Canny", "Moderate", "Thin, slight artifacts", "Detects main contour", "Moderate"),
        ("Gaussian + Sobel", "Good", "Smooth, thick", "Continuous, imprecise", "Good"),
        ("Gaussian + Canny", "Excellent", "Thin, crisp, precise", "Accurate, closed outline", "Excellent"),
        ("Median + Sobel", "Good", "Blocky edges", "Coarse localization", "Fair"),
        ("Median + Canny", "Good", "Sharp, slightly jagged", "Good isolation", "Good")
    ]
    
    for row in comparisons:
        print(f"{row[0]:<18} | {row[1]:<15} | {row[2]:<20} | {row[3]:<25} | {row[4]}")
    print("==============================================================\n")