import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import time
import os

# Create directory for outputs
os.makedirs('results', exist_ok=True)

def get_image():
    """Loads a sample image or creates a synthetic one if not found."""
    img = cv2.imread('sample_image.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("No 'sample_image.jpg' found. Generating a synthetic test image.")
        img = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (250, 250), 255, -1)
        cv2.circle(img, (150, 150), 50, 0, -1)
        # Add some texture for edge detection testing
        for i in range(0, 300, 20):
            cv2.line(img, (i, 0), (i, 300), 128, 1)
    return img

def add_noise(image):
    # Gaussian
    gauss = np.random.normal(0, 25, image.shape).astype('uint8')
    img_gauss = cv2.add(image, gauss)
    # Salt and Pepper
    img_sp = np.copy(image)
    prob = 0.05
    salt = np.random.rand(*image.shape) < prob / 2
    pepper = np.random.rand(*image.shape) < prob / 2
    img_sp[salt] = 255
    img_sp[pepper] = 0
    return img_gauss, img_sp

# --- Task 1: Comparative Edge Detection ---
def task1_edges(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    
    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_mag = cv2.addWeighted(cv2.filter2D(img, -1, kx), 0.5, cv2.filter2D(img, -1, ky), 0.5, 0)
    
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    log = cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_64F)
    canny = cv2.Canny(img, 100, 200)
    
    images = [img, np.uint8(np.absolute(sobel_mag)), prewitt_mag, np.uint8(np.absolute(laplacian)), np.uint8(np.absolute(log)), canny]
    titles = ['Original', 'Sobel', 'Prewitt', 'Laplacian', 'LoG', 'Canny']
    
    plt.figure(figsize=(18, 4))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.savefig('results/Task1_Edge_Comparison.png')
    plt.close()

# --- Task 2: Noise Effects ---
def task2_noise(img, img_gauss, img_sp):
    gauss_filtered = cv2.GaussianBlur(img_gauss, (5, 5), 0)
    median_filtered = cv2.medianBlur(img_sp, 5)
    
    def apply_sobel(image):
        return np.uint8(np.absolute(cv2.magnitude(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3))))

    images = [apply_sobel(img), apply_sobel(img_gauss), apply_sobel(img_sp), apply_sobel(gauss_filtered), apply_sobel(median_filtered)]
    titles = ['Orig Edge', 'Gauss Noise', 'S&P Noise', 'Gauss Filtered', 'Median Filtered']
    
    plt.figure(figsize=(15, 4))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.savefig('results/Task2_Noise_Effects.png')
    plt.close()

# --- Task 3: Canny Parameters ---
def task3_canny(img):
    images = [cv2.Canny(img, 30, 100), cv2.Canny(img, 50, 150), cv2.Canny(img, 100, 200)]
    titles = ['Low=30, High=100', 'Low=50, High=150', 'Low=100, High=200']
    
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.savefig('results/Task3_Canny_Parameters.png')
    plt.close()

# --- Task 4, 5, 6: Classification ---
def task4_5_6_classification():
    # Generating 150 synthetic 20x20 images for demonstration
    np.random.seed(42)
    X = np.random.rand(150, 400) 
    y = np.random.randint(0, 3, 150)
    
    X_raw = X
    X_filtered = cv2.GaussianBlur(X.reshape(-1, 20, 20), (3, 3), 0).reshape(-1, 400)
    X_edge = np.array([cv2.Canny((i * 255).astype(np.uint8), 100, 200) for i in X.reshape(-1, 20, 20)]).reshape(-1, 400)
    
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
    names = ['Raw', 'Filtered', 'Edge']
    
    for name, data in zip(names, [X_raw, X_filtered, X_edge]):
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        
        start = time.time()
        y_pred = clf.predict(X_test)
        inf_time = (time.time() - start) / len(X_test) * 1000
        
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['Recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        metrics['F1'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        
        # Save CM
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
        disp.plot(cmap='Blues')
        plt.title(f'{name} Images CM')
        plt.savefig(f'results/Task6_CM_{name}.png')
        plt.close()
        
        print(f"{name} -> Acc: {metrics['Accuracy'][-1]:.2f}, Train(s): {train_time:.4f}, Inf(ms): {inf_time:.4f}")

    # Plot Bar Chart
    x = np.arange(3)
    width = 0.2
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5*width, metrics['Accuracy'], width, label='Accuracy')
    plt.bar(x - 0.5*width, metrics['Precision'], width, label='Precision')
    plt.bar(x + 0.5*width, metrics['Recall'], width, label='Recall')
    plt.bar(x + 1.5*width, metrics['F1'], width, label='F1-Score')
    plt.xticks(x, names)
    plt.legend()
    plt.title('Performance Comparison')
    plt.savefig('results/Task6_BarChart.png')
    plt.close()

if __name__ == "__main__":
    img = get_image()
    img_gauss, img_sp = add_noise(img)
    
    print("Running Task 1: Edge Detectors...")
    task1_edges(img)
    print("Running Task 2: Noise Effects...")
    task2_noise(img, img_gauss, img_sp)
    print("Running Task 3: Canny Params...")
    task3_canny(img)
    print("Running Task 4-6: Classification Pipeline...")
    task4_5_6_classification()
    print("\nAll tasks done! Check the 'results' folder.")