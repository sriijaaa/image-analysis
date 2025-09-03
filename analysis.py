import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json

class ImprovedWeddingImageQualityChecker:
    def __init__(self):
        """Enhanced Image Quality Assessment for Wedding Photography - Applied to Bounding Boxes"""
        # Standard thresholds
        self.standard_brightness_range = (60, 180)

        # Blur thresholds (strict enforcement)
        self.blur_poor_threshold = 150     # Below this → POOR
        self.blur_acceptable_threshold = 300  # Between 150–300 → ACCEPTABLE
        self.good_sharpness_threshold = 350   # Above this → GOOD
        self.excellent_sharpness_threshold = 500  # Above this → EXCELLENT

    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Use Laplacian variance only (reliable blur detector)"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)

    def _calculate_local_contrast(self, gray: np.ndarray) -> float:
        kernel_size = min(gray.shape[0]//8, gray.shape[1]//8, 50)
        kernel_size = max(kernel_size, 15)
        local_stds = []
        step = kernel_size // 2

        for i in range(0, gray.shape[0] - kernel_size, step):
            for j in range(0, gray.shape[1] - kernel_size, step):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                local_stds.append(np.std(patch))

        return np.mean(local_stds) if local_stds else np.std(gray)

    def _detect_bbox_style(self, roi: np.ndarray) -> str:
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:,:,0]
        brightness = np.mean(Y)

        hist = cv2.calcHist([Y], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)

        dark_pixels = np.sum(hist[:60])
        bright_pixels = np.sum(hist[200:])

        if (brightness < 80 and dark_pixels > 0.4) or (dark_pixels > 0.6):
            return "low-key"
        elif brightness > 180 and bright_pixels > 0.3:
            return "high-key"
        else:
            return "standard"

    def _assess_bbox_artistic_quality(
        self, roi: np.ndarray, style: str, bbox_size: Tuple[int, int]
    ) -> Tuple[str, List[str]]:
        if roi.size == 0:
            return "FAIL", ["Invalid bounding box region"]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        Y_roi, Cr_roi, Cb_roi = cv2.split(ycrcb_roi)

        issues = []
        positives = []

        brightness = np.mean(Y_roi)
        sharpness = self._calculate_sharpness(gray_roi)
        global_contrast = np.std(Y_roi)
        local_contrast = self._calculate_local_contrast(gray_roi)

        w, h = bbox_size
        total_pixels = w * h
        overexposed = np.sum(Y_roi > 250) / total_pixels
        underexposed = np.sum(Y_roi < 5) / total_pixels

        # --- 🔑 STRICT BLUR RULES ---
        if sharpness < self.blur_poor_threshold:
            return "POOR", ["Too blurry - automatically POOR"]
        elif sharpness < self.blur_acceptable_threshold:
            return "ACCEPTABLE", ["Slight blur detected - capped at ACCEPTABLE"]

        # --- Exposure/style-based rules ---
        if style == "low-key":
            if underexposed > 0.5 or brightness < 10:
                return "FAIL", ["Person lost in shadows - too dark"]
            if local_contrast > 25:
                positives.append("Strong dramatic contrast creates mood")

        elif style == "high-key":
            if overexposed > 0.4:
                return "FAIL", ["Person details lost in highlights"]
            if brightness > 150 and global_contrast > 25:
                positives.append("Bright, airy high-key style")

        else:  # standard
            if brightness < self.standard_brightness_range[0]:
                return "FAIL", ["Person too dark - underexposed"]
            elif brightness > self.standard_brightness_range[1]:
                return "FAIL", ["Person too bright - overexposed"]

        # --- Sharpness-based positives ---
        if sharpness > self.excellent_sharpness_threshold:
            positives.append("Excellent sharp focus on person")
            return "EXCELLENT", positives
        elif sharpness > self.good_sharpness_threshold:
            positives.append("Good focus quality")
            return "GOOD", positives

        return "ACCEPTABLE", ["Meets minimum standards"]

    def _get_style_description(self, style: str) -> str:
        return {
            "low-key": "Dramatic/Artistic Low-Key",
            "high-key": "Bright/Airy High-Key",
            "standard": "Standard Exposure"
        }.get(style, "Unknown")

    def analyze_bounding_boxes(self, image_path: str, bbox_data: str, show_details: bool = True) -> str:
        try:
            img = cv2.imread(image_path)
            if img is None:
                return f"❌ Error: Could not read '{image_path}'"

            try:
                bbox_dict = json.loads(bbox_data)
                filename = list(bbox_dict.keys())[0]
                bboxes = bbox_dict[filename]
            except:
                return "❌ Error: Invalid JSON format for bounding boxes"

            overall_style = self._detect_bbox_style(img)
            emoji_map = {'EXCELLENT':'🌟','GOOD':'✅','ACCEPTABLE':'⚠️','POOR':'❌','FAIL':'❌'}

            result = f"=== ENHANCED WEDDING PHOTO ANALYSIS: {filename} ===\n"
            result += f"Overall Image Style: {self._get_style_description(overall_style)}\n"
            result += f"Total People Detected: {len(bboxes)}\n\n"

            verdicts = []
            for i, bbox_info in enumerate(bboxes, 1):
                coords = bbox_info["coordinates"]
                confidence = bbox_info["confidence"]
                x, y, x2, y2 = coords
                w, h = x2 - x, y2 - y
                roi = img[y:y+h, x:x+w]

                if roi.size == 0:
                    verdict, explanations = "FAIL", ["Invalid bounding box"]
                else:
                    bbox_style = self._detect_bbox_style(roi)
                    verdict, explanations = self._assess_bbox_artistic_quality(roi, bbox_style, (w, h))

                verdicts.append(verdict)

                result += f"👤 PERSON {i} {emoji_map[verdict]} {verdict}\n"
                result += f"   Confidence: {confidence:.2f} | Size: {w}x{h}px\n"
                if roi.size > 0:
                    result += f"   Style: {self._get_style_description(bbox_style)}\n"
                result += f"   Assessment: {explanations[0]}\n\n"

            # --- Summary ---
            excellent_count = verdicts.count('EXCELLENT')
            good_count = verdicts.count('GOOD')
            acceptable_count = verdicts.count('ACCEPTABLE')
            poor_count = verdicts.count('POOR')
            fail_count = verdicts.count('FAIL')

            result += "="*60 + "\n📊 DETAILED SUMMARY:\n"
            result += f"🌟 {excellent_count} Excellent | ✅ {good_count} Good | ⚠️ {acceptable_count} Acceptable | ❌ {poor_count} Poor | ❌ {fail_count} Failed\n\n"

            # --- Overall ---
            if "POOR" in verdicts or "ACCEPTABLE" in verdicts:
                result += "⚠️ Some blur detected → Overall capped at ACCEPTABLE."
            else:
                result += "👍 All people sharp and clear."

            return result
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Convenience
def analyze_people_comprehensive(image_path: str, bbox_json: str) -> str:
    checker = ImprovedWeddingImageQualityChecker()
    return checker.analyze_bounding_boxes(image_path, bbox_json, show_details=False)

def analyze_people_detailed(image_path: str, bbox_json: str) -> str:
    checker = ImprovedWeddingImageQualityChecker()
    return checker.analyze_bounding_boxes(image_path, bbox_json, show_details=True)

# ==== USAGE ====
if __name__ == "__main__":
    image_file = r"C:\Users\HP\OneDrive\Desktop\photo\H Wedding 6 CP_no crop\SBCL6132.JPG"
    bbox_data = '''{ "SBCL6132.JPG": [ { "coordinates": [ 244, 0, 718, 917 ], "confidence": 0.76 }, { "coordinates": [ 95, 394, 683, 1077 ], "confidence": 0.7 }, { "coordinates": [ 0, 220, 168, 403 ], "confidence": 0.69 }, { "coordinates": [ 526, 381, 643, 638 ], "confidence": 0.58 } ]}'''


    print("\n" + "="*60)
    print("=== DETAILED ===")
    print(analyze_people_detailed(image_file, bbox_data))
