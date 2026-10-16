"""
Lab 06 - Landmark Recognition Using SIFT Algorithm
BAI-07 Computer Vision

Tasks covered:
  1. Landmark image database
  2. SIFT feature extraction on reference images
  3. Query image processing
  4. Feature matching + landmark recognition (BF + KNN + Lowe's ratio test)
  5. Landmark localization (homography / RANSAC)
  6. Performance evaluation table

Usage:
  python lab06_landmark_recognition.py

  By default the script:
    - Downloads sample reference images from Wikimedia Commons.
    - Lets you pick any query image interactively OR uses the bundled test images.
    - Saves all output figures to ./output/

Requirements:
  pip install opencv-python opencv-contrib-python numpy requests Pillow
"""

import os
import time
import urllib.request
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_DIR     = os.path.join(BASE_DIR, "database")
QUERY_DIR  = os.path.join(BASE_DIR, "query_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(QUERY_DIR,  exist_ok=True)

LOWE_RATIO      = 0.75   # Lowe's ratio test threshold
MIN_MATCH_COUNT = 10     # minimum reliable matches to accept a landmark
RANSAC_THRESH   = 5.0    # RANSAC reprojection threshold (pixels)

# Landmark name  ->  list of (filename, public-domain image URL)
LANDMARK_IMAGES = {
    "Eiffel Tower": [
        ("eiffel1.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/"
         "Tour_Eiffel_Wikimedia_Commons.jpg/640px-Tour_Eiffel_Wikimedia_Commons.jpg"),
        ("eiffel2.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/"
         "Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg/640px-Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"),
    ],
    "Taj Mahal": [
        ("taj1.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/"
         "Taj_Mahal_%28Edited%29.jpeg/640px-Taj_Mahal_%28Edited%29.jpeg"),
        ("taj2.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/"
         "Taj_Mahal%2C_Agra%2C_India_edit3.jpg/640px-Taj_Mahal%2C_Agra%2C_India_edit3.jpg"),
    ],
    "Great Wall": [
        ("wall1.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/"
         "The_Great_Wall_of_China_at_Jinshanling-edit.jpg/"
         "640px-The_Great_Wall_of_China_at_Jinshanling-edit.jpg"),
        ("wall2.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/"
         "Great_Wall_of_China_July_2006.jpg/640px-Great_Wall_of_China_July_2006.jpg"),
    ],
    "Statue of Liberty": [
        ("liberty1.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/"
         "Statue_of_Liberty_7.jpg/480px-Statue_of_Liberty_7.jpg"),
        ("liberty2.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
         "Statue_of_Liberty%2C_NY.jpg/480px-Statue_of_Liberty%2C_NY.jpg"),
    ],
    "Colosseum": [
        ("colosseum1.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/"
         "Colosseo_2020.jpg/640px-Colosseo_2020.jpg"),
        ("colosseum2.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/"
         "Colosseum_in_Rome%2C_Italy_-_April_2007.jpg/"
         "640px-Colosseum_in_Rome%2C_Italy_-_April_2007.jpg"),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: Download / organise reference database
# ─────────────────────────────────────────────────────────────────────────────

def download_database():
    """Download reference images if not already present."""
    print("\n[Task 1] Setting up landmark database …")
    headers = {"User-Agent": "Mozilla/5.0 (LandmarkRecognitionLab/1.0)"}

    for landmark, images in LANDMARK_IMAGES.items():
        folder_name = landmark.lower().replace(" ", "_")
        folder = os.path.join(DB_DIR, folder_name)
        os.makedirs(folder, exist_ok=True)

        for filename, url in images:
            dest = os.path.join(folder, filename)
            if not os.path.exists(dest):
                print(f"  Downloading {landmark}/{filename} …", end=" ", flush=True)
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=30) as r, \
                         open(dest, "wb") as f:
                        f.write(r.read())
                    print("OK")
                except Exception as e:
                    print(f"FAILED ({e})")
            else:
                print(f"  {landmark}/{filename} already exists, skipping.")
    print("[Task 1] Database ready.\n")


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: SIFT feature extraction on reference images
# ─────────────────────────────────────────────────────────────────────────────

def build_descriptor_database(sift):
    """
    Load every reference image, extract SIFT descriptors, and return:
      database = { landmark_name: [descriptors, ...] }
    Also saves keypoint-visualisation images to ./output/
    """
    print("[Task 2] Extracting SIFT descriptors from reference images …")
    database = {}

    for landmark, images in LANDMARK_IMAGES.items():
        folder_name = landmark.lower().replace(" ", "_")
        folder = os.path.join(DB_DIR, folder_name)
        database[landmark] = []

        for filename, _ in images:
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"  WARNING: Could not read {path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kps, descs = sift.detectAndCompute(gray, None)

            if descs is None:
                continue
            database[landmark].append(descs)

            # Visualise & save keypoints
            kp_img = cv2.drawKeypoints(
                img, kps, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            tag = filename.replace(".jpg", "")
            out_path = os.path.join(OUTPUT_DIR, f"kp_ref_{folder_name}_{tag}.jpg")
            cv2.imwrite(out_path, kp_img)
            print(f"  {landmark}/{filename}: {len(kps)} keypoints  → {out_path}")

    print("[Task 2] Done.\n")
    return database


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: Query image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_query(query_path, sift):
    """Load query image, extract SIFT features, and return (img, gray, kps, descs)."""
    img = cv2.imread(query_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load query image: {query_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, descs = sift.detectAndCompute(gray, None)

    kp_img = cv2.drawKeypoints(
        img, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    name = os.path.splitext(os.path.basename(query_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"kp_query_{name}.jpg")
    cv2.imwrite(out_path, kp_img)
    print(f"  Query keypoints: {len(kps)}  → {out_path}")

    return img, gray, kps, descs


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: Feature matching and landmark recognition
# ─────────────────────────────────────────────────────────────────────────────

def match_against_database(query_descs, database, bf):
    """
    Compare query descriptors against every reference descriptor set.
    Returns sorted list of (landmark, reliable_match_count).
    """
    scores = {}
    for landmark, desc_list in database.items():
        total_good = 0
        for ref_descs in desc_list:
            if ref_descs is None or len(ref_descs) < 2:
                continue
            matches = bf.knnMatch(query_descs, ref_descs, k=2)
            # Lowe's ratio test
            good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
            total_good += len(good)
        scores[landmark] = total_good

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def recognize_landmark(query_path, sift, bf, database):
    """Full recognition pipeline for one query image. Returns result dict."""
    print(f"\n[Task 3+4] Processing query: {query_path}")
    t0 = time.time()

    query_img, query_gray, query_kps, query_descs = process_query(query_path, sift)
    n_kps = len(query_kps)

    if query_descs is None or len(query_descs) < 2:
        return {
            "query": query_path,
            "n_keypoints": n_kps,
            "predicted": "UNKNOWN",
            "n_matches": 0,
            "recognized": False,
            "time_s": round(time.time() - t0, 3),
        }

    ranked = match_against_database(query_descs, database, bf)

    best_landmark, best_score = ranked[0]
    recognized = best_score >= MIN_MATCH_COUNT

    print(f"  Ranked candidates:")
    for name, score in ranked:
        print(f"    {name:25s}: {score} reliable matches")

    if recognized:
        print(f"  ✓ Recognized: {best_landmark} ({best_score} matches)")
    else:
        print(f"  ✗ Not recognized (best={best_score} < threshold={MIN_MATCH_COUNT})")

    elapsed = round(time.time() - t0, 3)
    return {
        "query": query_path,
        "n_keypoints": n_kps,
        "predicted": best_landmark if recognized else "UNKNOWN",
        "n_matches": best_score,
        "recognized": recognized,
        "ranked": ranked,
        "time_s": elapsed,
        "query_img": query_img,
        "query_gray": query_gray,
        "query_kps": query_kps,
        "query_descs": query_descs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5: Landmark localization (homography + RANSAC)
# ─────────────────────────────────────────────────────────────────────────────

def localize_landmark(result, sift, bf, database):
    """
    Find the best reference image for the predicted landmark and draw
    the homography polygon on the query image.
    """
    if not result.get("recognized"):
        print("  [Task 5] Skipping localization (landmark not recognized).")
        return

    predicted   = result["predicted"]
    query_img   = result["query_img"]
    query_descs = result["query_descs"]
    query_kps   = result["query_kps"]

    folder_name = predicted.lower().replace(" ", "_")
    folder      = os.path.join(DB_DIR, folder_name)

    best_ref_img   = None
    best_ref_kps   = None
    best_good      = []
    best_ref_descs = None

    # Find the single reference image with the most good matches
    for filename, _ in LANDMARK_IMAGES[predicted]:
        path = os.path.join(folder, filename)
        ref_img = cv2.imread(path)
        if ref_img is None:
            continue
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_kps, ref_descs = sift.detectAndCompute(ref_gray, None)
        if ref_descs is None:
            continue

        matches = bf.knnMatch(query_descs, ref_descs, k=2)
        good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]

        if len(good) > len(best_good):
            best_good      = good
            best_ref_img   = ref_img
            best_ref_kps   = ref_kps
            best_ref_descs = ref_descs

    if best_ref_img is None or len(best_good) < 4:
        print("  [Task 5] Not enough matches for homography.")
        return

    # Draw match image
    match_img = cv2.drawMatchesKnn(
        best_ref_img, best_ref_kps,
        query_img, query_kps,
        [[m] for m in best_good], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    name     = os.path.splitext(os.path.basename(result["query"]))[0]
    out_path = os.path.join(OUTPUT_DIR, f"matches_{name}.jpg")
    cv2.imwrite(out_path, match_img)
    print(f"  Match image → {out_path}")

    # Homography estimation with RANSAC
    src_pts = np.float32(
        [best_ref_kps[m.queryIdx].pt for m in best_good]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [query_kps[m.trainIdx].pt for m in best_good]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts,
                                  cv2.RANSAC, RANSAC_THRESH)
    if H is None:
        print("  [Task 5] Homography computation failed.")
        return

    inliers = int(mask.sum())
    print(f"  [Task 5] Reliable matches: {len(best_good)}  |  RANSAC inliers: {inliers}")

    if inliers < 4:
        print("  [Task 5] Too few inliers — geometric verification unreliable.")
        return

    # Project reference image corners onto query image
    h, w = best_ref_img.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)

    # Draw polygon
    vis = query_img.copy()
    cv2.polylines(vis, [np.int32(projected)], True, (0, 255, 0), 3)
    label = f"{predicted}  ({len(best_good)} matches, {inliers} inliers)"
    cv2.putText(vis, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    out_path = os.path.join(OUTPUT_DIR, f"localized_{name}.jpg")
    cv2.imwrite(out_path, vis)
    print(f"  Localized image → {out_path}")

    result["n_inliers"] = inliers


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6: Performance evaluation – generate test images automatically
# ─────────────────────────────────────────────────────────────────────────────

def make_test_variants(source_path, tag):
    """
    Given a reference image, produce 6 test variants:
      original, rotated, resized, dark, viewpoint-shift, unknown
    Returns list of (variant_name, variant_path).
    """
    img = cv2.imread(source_path)
    if img is None:
        return []

    variants = []
    h, w = img.shape[:2]

    # 1. Original
    p = os.path.join(QUERY_DIR, f"{tag}_original.jpg")
    cv2.imwrite(p, img)
    variants.append(("Original", p))

    # 2. Rotated 45°
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)
    rot = cv2.warpAffine(img, M, (w, h))
    p = os.path.join(QUERY_DIR, f"{tag}_rotated.jpg")
    cv2.imwrite(p, rot)
    variants.append(("Rotated 45°", p))

    # 3. Resized (50 %)
    small = cv2.resize(img, (w // 2, h // 2))
    p = os.path.join(QUERY_DIR, f"{tag}_resized.jpg")
    cv2.imwrite(p, small)
    variants.append(("Resized 50%", p))

    # 4. Reduced brightness
    dark = np.clip(img.astype(np.int32) - 80, 0, 255).astype(np.uint8)
    p = os.path.join(QUERY_DIR, f"{tag}_dark.jpg")
    cv2.imwrite(p, dark)
    variants.append(("Dark (-80)", p))

    # 5. Viewpoint shift (perspective warp)
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts2 = np.float32([[w*0.1, 0], [w*0.9, 0], [w, h], [0, h]])
    M2   = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M2, (w, h))
    p = os.path.join(QUERY_DIR, f"{tag}_viewpoint.jpg")
    cv2.imwrite(p, warped)
    variants.append(("Viewpoint shift", p))

    return variants


def performance_evaluation(sift, bf, database):
    """
    Task 6: Run all test variants for the first reference image of each landmark
    + an 'unknown' image (solid colour).  Print and return a results table.
    """
    print("\n[Task 6] Performance Evaluation\n")

    rows = []
    # Build per-landmark test cases
    for landmark, images in LANDMARK_IMAGES.items():
        folder_name = landmark.lower().replace(" ", "_")
        src = os.path.join(DB_DIR, folder_name, images[0][0])
        tag = folder_name

        variants = make_test_variants(src, tag)

        # Also add an 'unknown' test: a random noise image
        unknown_path = os.path.join(QUERY_DIR, f"{tag}_unknown.jpg")
        noise = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite(unknown_path, noise)
        variants.append(("Unknown (noise)", unknown_path))

        for variant_name, vpath in variants:
            result = recognize_landmark(vpath, sift, bf, database)
            localize_landmark(result, sift, bf, database)
            rows.append({
                "Landmark":   landmark,
                "Variant":    variant_name,
                "Keypoints":  result["n_keypoints"],
                "Matches":    result["n_matches"],
                "Predicted":  result["predicted"],
                "Recognized": "Yes" if result["recognized"] else "No",
                "Time (s)":   result["time_s"],
            })

    # Print table
    print("\n" + "=" * 95)
    print(f"{'Landmark':<22} {'Variant':<20} {'KPs':>5} {'Matches':>8} "
          f"{'Predicted':<22} {'Recog.':>7} {'Time(s)':>8}")
    print("=" * 95)
    for r in rows:
        print(f"{r['Landmark']:<22} {r['Variant']:<20} {r['Keypoints']:>5} "
              f"{r['Matches']:>8} {r['Predicted']:<22} {r['Recognized']:>7} "
              f"{r['Time (s)']:>8}")
    print("=" * 95)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Lab 06 – Landmark Recognition Using SIFT")
    print("=" * 60)

    # Initialise SIFT and BFMatcher
    sift = cv2.SIFT_create()
    bf   = cv2.BFMatcher(cv2.NORM_L2)   # Euclidean distance for SIFT

    # Task 1: Download database
    download_database()

    # Task 2: Build descriptor database
    database = build_descriptor_database(sift)

    # Task 6: Performance evaluation (includes Tasks 3, 4, 5 internally)
    rows = performance_evaluation(sift, bf, database)

    print(f"\nAll output images saved to: {OUTPUT_DIR}")
    print("Done.\n")


if __name__ == "__main__":
    main()
