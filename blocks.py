import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from dataclasses import dataclass
from enum import Enum

class ImageStyle(Enum):
    STANDARD = "standard"
    LOW_KEY = "low-key"
    HIGH_KEY = "high-key"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"

@dataclass
class BlockMetrics:
    """Comprehensive metrics for image block analysis"""
    brightness: float
    sharpness: float
    global_contrast: float
    local_contrast: float
    saturation: float
    overexposed_ratio: float
    underexposed_ratio: float
    near_overexposed_ratio: float
    noise_level: float
    edge_density: float
    color_harmony: float
    focus_quality: float
    histogram_spread: float

@dataclass
class QualityAssessment:
    """Quality assessment result for a block or entire image"""
    verdict: str
    score: float
    issues: List[str]

class FixedWeddingImageQualityChecker:
    def __init__(self):
        """
        Fixed Wedding Image Quality Assessment with strict brightness control
        """
        # FIXED: Much stricter thresholds for wedding photography
        self.thresholds = {
            'standard': {
                'brightness_range': (70, 160),    # Was (60, 200) - too lenient
                'sharpness_min': 120,
                'contrast_min': 30,
                'saturation_range': (300, 4000),
                'noise_max': 25,
                'focus_min': 0.15,
                'overexpose_threshold': 0.03,     # Was 0.15 - much stricter
                'highlight_warning': 240,         # New - check near-blown highlights
                'wedding_brightness_max': 150     # New - special wedding limit
            },
            'low-key': {
                'brightness_range': (20, 110),    # Was (20, 90)
                'sharpness_min': 100,
                'contrast_min': 20,
                'saturation_range': (200, 3000),
                'noise_max': 30,
                'focus_min': 0.12,
                'overexpose_threshold': 0.02,
                'highlight_warning': 235,
                'wedding_brightness_max': 130
            },
            'high-key': {
                'brightness_range': (120, 180),   # Was (140, 240) - much stricter
                'sharpness_min': 110,
                'contrast_min': 15,
                'saturation_range': (250, 3500),
                'noise_max': 20,
                'focus_min': 0.13,
                'overexpose_threshold': 0.05,     # Still strict for high-key
                'highlight_warning': 245,
                'wedding_brightness_max': 170
            },
            'portrait': {
                'brightness_range': (75, 155),    # Was (70, 190) - stricter
                'sharpness_min': 150,
                'contrast_min': 35,
                'saturation_range': (400, 4500),
                'noise_max': 20,
                'focus_min': 0.18,
                'overexpose_threshold': 0.02,     # Very strict for portraits
                'highlight_warning': 238,
                'wedding_brightness_max': 145
            }
        }
        
        # Weight importance of different blocks for wedding photos
        self.block_weights = {
            "Top-Left": 0.2,      # Often background/environment
            "Top-Right": 0.2,     # Often background/environment  
            "Bottom-Left": 0.3,   # Often main subjects/faces
            "Bottom-Right": 0.3   # Often main subjects/faces
        }
        
    def _divide_image_into_blocks(self, img: np.ndarray) -> List[Tuple[np.ndarray, int, int, str]]:
        """Divide image into 4 blocks (2x2 grid)"""
        h, w = img.shape[:2]
        block_h = h // 2
        block_w = w // 2
        
        blocks = []
        block_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for i, (row, col) in enumerate(positions):
            y1 = row * block_h
            y2 = (row + 1) * block_h if row == 0 else h
            x1 = col * block_w
            x2 = (col + 1) * block_w if col == 0 else w
            
            block = img[y1:y2, x1:x2]
            blocks.append((block, row, col, block_names[i]))
        
        return blocks
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""
        if gray.size == 0:
            return 0
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def _calculate_local_contrast(self, gray: np.ndarray) -> float:
        """Calculate local contrast using standard deviation of local areas"""
        if gray.size == 0:
            return 0
            
        kernel_size = min(gray.shape[0]//4, gray.shape[1]//4, 30)
        kernel_size = max(kernel_size, 5)
        
        local_stds = []
        step = max(kernel_size // 2, 1)
        
        for i in range(0, gray.shape[0] - kernel_size + 1, step):
            for j in range(0, gray.shape[1] - kernel_size + 1, step):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                local_stds.append(np.std(patch))
        
        return np.mean(local_stds) if local_stds else np.std(gray)
    
    def _calculate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level using high-frequency content analysis"""
        if gray.size == 0:
            return 0
            
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        noise_map = cv2.absdiff(gray.astype(np.float32), blurred.astype(np.float32))
        return np.std(noise_map)
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection"""
        if gray.size == 0:
            return 0
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def _calculate_color_harmony(self, img: np.ndarray) -> float:
        """Calculate color harmony using HSV color distribution"""
        if img.size == 0:
            return 0
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
        hist_h = hist_h.flatten() / np.sum(hist_h)
        
        entropy = -np.sum(hist_h * np.log(hist_h + 1e-10))
        ideal_entropy = 4.5
        harmony_score = 1.0 - abs(entropy - ideal_entropy) / ideal_entropy
        
        return max(0, harmony_score) * 1000
    
    def _calculate_focus_quality(self, gray: np.ndarray) -> float:
        """Calculate focus quality using frequency domain analysis"""
        if gray.size == 0:
            return 0
            
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        low_freq_radius = min(h, w) // 8
        high_freq_start = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        center_dist = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        
        low_freq_mask = center_dist <= low_freq_radius
        high_freq_mask = center_dist >= high_freq_start
        
        low_freq_energy = np.mean(magnitude[low_freq_mask])
        high_freq_energy = np.mean(magnitude[high_freq_mask])
        
        focus_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        return min(focus_ratio, 1.0)
    
    def _calculate_histogram_spread(self, gray: np.ndarray) -> float:
        """Calculate histogram spread as measure of tonal range"""
        if gray.size == 0:
            return 0
            
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        cumsum = np.cumsum(hist)
        p5_idx = np.argmax(cumsum >= 0.05)
        p95_idx = np.argmax(cumsum >= 0.95)
        
        spread = (p95_idx - p5_idx) / 255.0
        return spread
    
    def _analyze_block_comprehensive(self, block: np.ndarray) -> BlockMetrics:
        """FIXED: Enhanced block analysis with strict highlight detection"""
        if block.size == 0:
            return BlockMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(block, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        
        # Basic metrics
        brightness = np.mean(Y)
        sharpness = self._calculate_sharpness(gray)
        global_contrast = np.std(Y)
        local_contrast = self._calculate_local_contrast(gray)
        saturation = (np.var(Cr) + np.var(Cb)) / 2
        
        # Advanced metrics
        noise_level = self._calculate_noise_level(gray)
        edge_density = self._calculate_edge_density(gray)
        color_harmony = self._calculate_color_harmony(block)
        focus_quality = self._calculate_focus_quality(gray)
        histogram_spread = self._calculate_histogram_spread(gray)
        
        # FIXED: Much more granular exposure analysis
        total_pixels = block.shape[0] * block.shape[1]
        
        # Multiple overexposure thresholds for precise detection
        severely_blown = np.sum(Y > 252) / total_pixels      # Pure white (total loss)
        blown_highlights = np.sum(Y > 245) / total_pixels    # Near pure white (severe loss)
        bright_highlights = np.sum(Y > 235) / total_pixels   # Very bright (detail at risk)
        underexposed = np.sum(Y < 5) / total_pixels          # Pure black
        
        # Use the most restrictive measurement for overexposed_ratio
        overexposed_ratio = severely_blown
        near_overexposed_ratio = blown_highlights - severely_blown
        
        return BlockMetrics(
            brightness=brightness,
            sharpness=sharpness,
            global_contrast=global_contrast,
            local_contrast=local_contrast,
            saturation=saturation,
            overexposed_ratio=overexposed_ratio,
            underexposed_ratio=underexposed,
            near_overexposed_ratio=near_overexposed_ratio,
            noise_level=noise_level,
            edge_density=edge_density,
            color_harmony=color_harmony,
            focus_quality=focus_quality,
            histogram_spread=histogram_spread
        )
    
    def _detect_image_style(self, img: np.ndarray) -> ImageStyle:
        """Simplified style detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:,:,0]
        
        brightness = np.mean(Y)
        
        hist = cv2.calcHist([Y], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        dark_pixels = np.sum(hist[:60])
        bright_pixels = np.sum(hist[200:])
        
        # Detect faces for portrait classification
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            face_coverage = sum(w * h for (x, y, w, h) in faces) / (img.shape[0] * img.shape[1])
            
            if face_coverage > 0.05:  # Significant face presence
                return ImageStyle.PORTRAIT
        except:
            pass  # Face detection failed, continue with other methods
        
        if (brightness < 80 and dark_pixels > 0.4) or (dark_pixels > 0.6):
            return ImageStyle.LOW_KEY
        elif brightness > 160 and bright_pixels > 0.3:
            return ImageStyle.HIGH_KEY
        else:
            return ImageStyle.STANDARD
    
    def _assess_block_quality_fixed(self, metrics: BlockMetrics, style: ImageStyle, 
                                   block_name: str) -> QualityAssessment:
        """FIXED: Much stricter quality assessment for wedding photography"""
        
        thresholds = self.thresholds.get(style.value, self.thresholds['standard'])
        
        scores = []
        issues = []
        
        # FIXED: Much stricter brightness assessment
        bright_min, bright_max = thresholds['brightness_range']
        wedding_max = thresholds['wedding_brightness_max']
        
        if bright_min <= metrics.brightness <= bright_max:
            if metrics.brightness <= wedding_max:
                scores.append(0.9)
            else:
                scores.append(0.6)  # Still within range but too bright for weddings
                issues.append("Too bright for wedding details")
        elif metrics.brightness > bright_max:
            # Much more aggressive penalty for overexposure
            severity = (metrics.brightness - bright_max) / bright_max
            if severity > 0.25:  # Severely overexposed
                scores.append(0.1)
                issues.append("Severely overexposed")
            elif severity > 0.15:  # Moderately overexposed  
                scores.append(0.3)
                issues.append("Overexposed")
            else:  # Slightly overexposed
                scores.append(0.5)
                issues.append("Slightly overexposed")
        else:  # Underexposed
            severity = (bright_min - metrics.brightness) / bright_min
            if severity > 0.4:
                scores.append(0.2)
                issues.append("Too dark")
            else:
                scores.append(0.6)
                issues.append("Slightly dark")
        
        # FIXED: Much stricter overexposure detection
        overexpose_threshold = thresholds['overexpose_threshold']
        if metrics.overexposed_ratio > overexpose_threshold:
            # Heavy penalty for any blown highlights
            penalty_factor = min(5.0, metrics.overexposed_ratio / overexpose_threshold)
            scores.append(max(0.1, 0.6 - penalty_factor * 0.1))
            issues.append(f"Blown highlights ({metrics.overexposed_ratio:.1%})")
        
        # NEW: Check for near-blown highlights (detail loss warning)
        if metrics.near_overexposed_ratio > 0.05:  # 5% near-blown is problematic
            scores.append(0.4)
            issues.append(f"Highlight detail at risk ({metrics.near_overexposed_ratio:.1%})")
        
        # NEW: Wedding-specific white dress preservation check
        if style in [ImageStyle.PORTRAIT, ImageStyle.STANDARD]:
            if metrics.brightness > wedding_max and (metrics.overexposed_ratio > 0.01 or metrics.near_overexposed_ratio > 0.03):
                scores.append(0.2)  # Heavy penalty for wedding dress detail loss
                issues.append("White dress/clothing detail lost")
        
        # Sharpness assessment
        sharpness_min = thresholds['sharpness_min']
        if metrics.sharpness > sharpness_min * 1.5:
            scores.append(1.0)
        elif metrics.sharpness > sharpness_min:
            scores.append(0.8)
        elif metrics.sharpness > sharpness_min * 0.7:
            scores.append(0.6)
        else:
            scores.append(0.3)
            issues.append("Too blurry")
        
        # Contrast assessment
        contrast_min = thresholds['contrast_min']
        if metrics.local_contrast > contrast_min * 1.2:
            scores.append(0.9)
        elif metrics.local_contrast > contrast_min:
            scores.append(0.8)
        else:
            scores.append(0.5)
            issues.append("Low contrast")
        
        # Noise assessment
        noise_max = thresholds['noise_max']
        if metrics.noise_level < noise_max * 0.5:
            scores.append(0.9)
        elif metrics.noise_level < noise_max:
            scores.append(0.7)
        else:
            scores.append(0.4)
            issues.append("High noise")
        
        # Color saturation assessment
        sat_min, sat_max = thresholds['saturation_range']
        if sat_min <= metrics.saturation <= sat_max:
            scores.append(0.8)
        elif metrics.saturation < sat_min:
            scores.append(0.5)
            issues.append("Colors washed out")
        else:
            scores.append(0.6)
            issues.append("Oversaturated")
        
        # Focus quality assessment
        focus_min = thresholds['focus_min']
        if metrics.focus_quality > focus_min * 1.5:
            scores.append(1.0)
        elif metrics.focus_quality > focus_min:
            scores.append(0.8)
        else:
            scores.append(0.5)
            issues.append("Poor focus quality")
        
        # Underexposure check
        if metrics.underexposed_ratio > 0.15:
            scores.append(0.4)
            issues.append(f"Shadow detail lost ({metrics.underexposed_ratio:.1%})")
        
        # Calculate final score
        final_score = np.mean(scores) if scores else 0.5
        
        # FIXED: Stricter verdict thresholds
        if final_score >= 0.85:
            verdict = "EXCELLENT"
        elif final_score >= 0.70:  # Was 0.75 - stricter
            verdict = "GOOD"
        elif final_score >= 0.55:  # Was 0.6 - stricter
            verdict = "ACCEPTABLE"
        elif final_score >= 0.35:  # Was 0.4 - stricter
            verdict = "POOR"
        else:
            verdict = "FAIL"
        
        return QualityAssessment(
            verdict=verdict,
            score=final_score,
            issues=issues
        )
    
    def _calculate_weighted_final_verdict(self, block_assessments: List[Tuple[str, QualityAssessment]], 
                                        block_names: List[str]) -> Tuple[str, float, List[str]]:
        """Calculate final verdict using weighted block importance"""
        
        total_weighted_score = 0
        total_weight = 0
        all_issues = []
        
        for (block_name, assessment) in block_assessments:
            weight = self.block_weights.get(block_name, 0.25)
            weighted_score = assessment.score * weight
            total_weighted_score += weighted_score
            total_weight += weight
            all_issues.extend(assessment.issues)
        
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Count verdicts by type
        verdict_counts = {}
        for _, assessment in block_assessments:
            verdict = assessment.verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        # FIXED: Stricter final verdict logic
        fail_count = verdict_counts.get('FAIL', 0)
        poor_count = verdict_counts.get('POOR', 0)
        excellent_count = verdict_counts.get('EXCELLENT', 0)
        
        # Any failed blocks severely impact the final verdict
        if fail_count >= 2:
            final_verdict = "FAIL"
        elif fail_count >= 1 or poor_count >= 2:
            final_verdict = "POOR"
        elif final_score >= 0.85 and excellent_count >= 2:
            final_verdict = "EXCELLENT"
        elif final_score >= 0.70:
            final_verdict = "GOOD"
        elif final_score >= 0.55:
            final_verdict = "ACCEPTABLE"
        elif final_score >= 0.35:
            final_verdict = "POOR"
        else:
            final_verdict = "FAIL"
        
        return final_verdict, final_score, list(set(all_issues))
    
    def analyze_wedding_image_fixed(self, image_path: str) -> str:
        """Fixed wedding image quality analysis with strict brightness control"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return f"Error: Could not read '{image_path}'"
            
            # Detect style
            style = self._detect_image_style(img)
            
            # Basic image info
            h, w = img.shape[:2]
            
            # Divide into blocks and analyze
            blocks = self._divide_image_into_blocks(img)
            block_assessments = []
            
            result_lines = []
            result_lines.append(f"IMAGE: {os.path.basename(image_path)}")
            result_lines.append(f"Resolution: {w}x{h} | Style: {style.value.title()}")
            result_lines.append("")
            result_lines.append("BLOCK ANALYSIS:")
            result_lines.append("-" * 40)
            
            # Analyze each block
            for i, (block, row, col, block_name) in enumerate(blocks, 1):
                metrics = self._analyze_block_comprehensive(block)
                assessment = self._assess_block_quality_fixed(metrics, style, block_name)
                block_assessments.append((block_name, assessment))
                
                emoji_map = {
                    'EXCELLENT': '✓✓', 'GOOD': '✓', 'ACCEPTABLE': '~',
                    'POOR': 'X', 'FAIL': 'XX'
                }
                
                result_lines.append(f"Block {i} - {block_name}: {emoji_map[assessment.verdict]} {assessment.verdict} (Score: {assessment.score:.2f})")
                result_lines.append(f"  Brightness: {metrics.brightness:.0f} | Sharpness: {metrics.sharpness:.0f} | Contrast: {metrics.local_contrast:.1f}")
                
                if metrics.overexposed_ratio > 0:
                    result_lines.append(f"  Blown highlights: {metrics.overexposed_ratio:.1%}")
                if metrics.near_overexposed_ratio > 0:
                    result_lines.append(f"  Near-blown highlights: {metrics.near_overexposed_ratio:.1%}")
                    
                if assessment.issues:
                    result_lines.append(f"  Issues: {', '.join(assessment.issues)}")
                result_lines.append("")
            
            # Calculate final verdict
            final_verdict, final_score, all_issues = self._calculate_weighted_final_verdict(
                block_assessments, [name for _, _, _, name in blocks]
            )
            
            emoji_map = {
                'EXCELLENT': '✓✓', 'GOOD': '✓', 'ACCEPTABLE': '~',
                'POOR': 'X', 'FAIL': 'XX'
            }
            
            result_lines.append("=" * 40)
            result_lines.append(f"FINAL VERDICT: {emoji_map[final_verdict]} {final_verdict}")
            result_lines.append(f"Overall Score: {final_score:.2f}/1.00")
            
            if all_issues:
                result_lines.append(f"Main Issues: {', '.join(list(set(all_issues))[:3])}")
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"

# Main function
def analyze_wedding_image_strict(image_path: str) -> str:
    """
    Analyze wedding image with strict brightness and exposure control
    """
    checker = FixedWeddingImageQualityChecker()
    return checker.analyze_wedding_image_fixed(image_path)

# Usage example
if __name__ == "__main__":
    # Single image analysis
    image_file = r"C:\Users\HP\OneDrive\Desktop\photo\rororollercoaster.jpg"
    
    print("=== IMAGE ANALYSIS ===")
    result = analyze_wedding_image_strict(image_file)
    print(result)