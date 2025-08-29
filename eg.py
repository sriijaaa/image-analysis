import cv2
import numpy as np

def evaluate_image_quality(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return "❌ Error: Could not read image."
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # 1. Brightness & Contrast
    # -------------------------
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)

    if mean_brightness < 50:
        brightness_eval = "Too Dark"
        brightness_score = 3
    elif mean_brightness > 200:
        brightness_eval = "Too Bright"
        brightness_score = 3
    else:
        brightness_eval = "Good Lighting"
        brightness_score = 9

    if contrast < 30:
        contrast_eval = "Low Contrast"
        contrast_score = 4
    elif contrast > 100:
        contrast_eval = "Too High Contrast"
        contrast_score = 5
    else:
        contrast_eval = "Good Contrast"
        contrast_score = 9

    # -------------------------
    # 2. Sharpness
    # -------------------------
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 100:
        sharpness_eval = "Blurry"
        sharpness_score = 3
    else:
        sharpness_eval = "Sharp"
        sharpness_score = 9

    # -------------------------
    # 3. Color Distribution
    # -------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    s_std = np.std(s)

    if s_std < 20:
        color_eval = "Dull Colors"
        color_score = 4
    elif s_std > 70:
        color_eval = "Over-saturated Colors"
        color_score = 5
    else:
        color_eval = "Balanced Colors"
        color_score = 9

    # -------------------------
    # Final Evaluation
    # -------------------------
    issues = []
    if brightness_eval != "Good Lighting":
        issues.append(brightness_eval)
    if contrast_eval != "Good Contrast":
        issues.append(contrast_eval)
    if sharpness_eval != "Sharp":
        issues.append(sharpness_eval)
    if color_eval != "Balanced Colors":
        issues.append(color_eval)

    # Average score
    final_score = np.mean([brightness_score, contrast_score, sharpness_score, color_score])

    if not issues:
        final_eval = f"✅ Good Image (Score: {final_score:.1f}/10) - Lighting, sharpness, and colors are well balanced."
    else:
        final_eval = f"⚠️ Image Needs Improvement (Score: {final_score:.1f}/10) - Issues: {', '.join(issues)}"

    return final_eval


# -------------------------
# Example Usage
# -------------------------
print(evaluate_image_quality(r"C:\Users\HP\OneDrive\Desktop\photo\1503048941_image1800.jpg"))
