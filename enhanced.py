import cv2
import numpy as np
from typing import List, Tuple, Dict

class WeddingImageQualityChecker:
    def __init__(self):
        """Enhanced Image Quality Assessment for Wedding Photography"""
        # Standard thresholds
        self.standard_brightness_range = (50, 200)
        self.sharpness_threshold = 100
        self.contrast_threshold = 25
        
        # Low-key/artistic thresholds
        self.lowkey_brightness_range = (15, 80)
        self.lowkey_contrast_threshold = 15
        self.lowkey_sharpness_threshold = 80
        
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def _calculate_local_contrast(self, gray: np.ndarray) -> float:
        """Calculate local contrast using standard deviation of local areas"""
        # Use smaller kernel for local contrast measurement
        kernel_size = min(gray.shape[0]//8, gray.shape[1]//8, 50)
        kernel_size = max(kernel_size, 15)  # Minimum kernel size
        
        # Calculate local standard deviations
        local_stds = []
        step = kernel_size // 2
        
        for i in range(0, gray.shape[0] - kernel_size, step):
            for j in range(0, gray.shape[1] - kernel_size, step):
                patch = gray[i:i+kernel_size, j:j+kernel_size]
                local_stds.append(np.std(patch))
        
        return np.mean(local_stds) if local_stds else np.std(gray)
    
    def _detect_image_style(self, img: np.ndarray) -> str:
        """Detect if image is standard, low-key artistic, or high-key"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:,:,0]
        
        brightness = np.mean(Y)
        
        # Calculate histogram for distribution analysis
        hist = cv2.calcHist([Y], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        # Check for low-key characteristics
        dark_pixels = np.sum(hist[:60])  # Pixels in lower 25% of brightness
        bright_pixels = np.sum(hist[200:])  # Pixels in upper 22% of brightness
        
        # Low-key detection (dramatic/artistic lighting)
        if (brightness < 80 and dark_pixels > 0.4) or (dark_pixels > 0.6):
            return "low-key"
        
        # High-key detection (very bright/overexposed look)
        elif brightness > 180 and bright_pixels > 0.3:
            return "high-key"
        
        # Standard photography
        else:
            return "standard"
    
    def _assess_artistic_quality(self, img: np.ndarray, style: str) -> Tuple[str, List[str]]:
        """Assess quality based on detected style"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        
        issues = []
        positives = []
        
        # Basic measurements
        brightness = np.mean(Y)
        sharpness = self._calculate_sharpness(gray)
        global_contrast = np.std(Y)
        local_contrast = self._calculate_local_contrast(gray)
        saturation = (np.var(Cr) + np.var(Cb)) / 2
        
        total_pixels = img.shape[0] * img.shape[1]
        overexposed = np.sum(Y > 250) / total_pixels
        underexposed = np.sum(Y < 5) / total_pixels
        
        if style == "low-key":
            # Low-key artistic assessment
            
            # Critical failures for low-key
            if sharpness < self.lowkey_sharpness_threshold * 0.3:
                return "FAIL", ["Too blurry - even artistic images need sharp focus in key areas"]
            elif underexposed > 0.4:
                return "FAIL", ["Too much detail lost in shadows - even dramatic lighting needs some detail"]
            
            # Positive aspects of low-key
            if local_contrast > 20:
                positives.append("Strong dramatic contrast creates mood and depth")
            if sharpness > self.lowkey_sharpness_threshold:
                positives.append("Sharp focus where it matters")
            if brightness < 60 and local_contrast > 15:
                positives.append("Excellent low-key lighting technique")
            
            # Issues for low-key
            if local_contrast < self.lowkey_contrast_threshold:
                issues.append("Lacks sufficient contrast for dramatic effect")
            elif overexposed > 0.1:
                issues.append("Some highlights are blown out")
            
        elif style == "high-key":
            # High-key assessment
            if sharpness < self.sharpness_threshold * 0.4:
                return "FAIL", ["Too blurry for high-key style"]
            elif overexposed > 0.3:
                return "FAIL", ["Too much detail lost in highlights"]
            
            if brightness > 160 and global_contrast > 20:
                positives.append("Bright, airy high-key style well executed")
            if sharpness > self.sharpness_threshold:
                positives.append("Crisp focus maintains detail in bright exposure")
                
        else:
            # Standard photography assessment
            
            # Critical failures
            if brightness < self.standard_brightness_range[0]:
                return "FAIL", ["Too dark - image is underexposed"]
            elif brightness > self.standard_brightness_range[1]:
                return "FAIL", ["Too bright - image is overexposed"] 
            elif sharpness < self.sharpness_threshold * 0.3:
                return "FAIL", ["Too blurry - lacks sharp focus"]
            
            # Standard quality checks
            if saturation < 200:
                issues.append("Colors appear washed out")
            elif saturation > 5000:
                issues.append("Colors are oversaturated")
                
            if global_contrast < self.contrast_threshold:
                issues.append("Low contrast makes image appear flat")
                
            if overexposed > 0.15:
                issues.append("Bright areas are overexposed")
            elif underexposed > 0.2:
                issues.append("Dark areas are underexposed")
        
        # Determine final verdict
        if positives and len(issues) <= 1:
            if sharpness > self.sharpness_threshold * 1.5:
                return "EXCELLENT", positives + (issues if issues else ["Minor technical perfection"])
            else:
                return "GOOD", positives + issues
        elif len(issues) == 0 or (len(issues) <= 1 and style == "low-key"):
            return "GOOD", positives if positives else ["Technically sound image"]
        elif len(issues) <= 2:
            return "ACCEPTABLE", issues
        else:
            return "POOR", issues
    
    def _get_style_description(self, style: str) -> str:
        """Get description of detected style"""
        descriptions = {
            "low-key": "Dramatic/Artistic Low-Key",
            "high-key": "Bright/Airy High-Key", 
            "standard": "Standard Exposure"
        }
        return descriptions.get(style, "Unknown")
    
    def check_quality(self, image_path: str, show_details: bool = False) -> str:
        """
        Check image quality with style-aware assessment
        
        Args:
            image_path: Path to the image file
            show_details: Whether to show technical metrics and style info
            
        Returns:
            Quality assessment string
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return f"❌ Error: Could not read '{image_path}'"
            
            # Detect style and assess accordingly
            style = self._detect_image_style(img)
            verdict, explanations = self._assess_artistic_quality(img, style)
            
            # Format output
            emoji_map = {
                'EXCELLENT': '🌟',
                'GOOD': '✅', 
                'ACCEPTABLE': '✅',
                'POOR': '⚠️',
                'FAIL': '❌'
            }
            
            style_desc = self._get_style_description(style)
            result = f"{emoji_map[verdict]} {verdict} ({style_desc}) - {explanations[0]}"
            
            if len(explanations) > 1:
                result += "\nAdditional notes:"
                for explanation in explanations[1:]:
                    result += f"\n  • {explanation}"
            
            if show_details:
                # Add technical metrics
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                Y = ycrcb[:,:,0]
                
                brightness = np.mean(Y)
                sharpness = self._calculate_sharpness(gray)
                global_contrast = np.std(Y)
                local_contrast = self._calculate_local_contrast(gray)
                
                result += f"\n\nTechnical metrics:"
                result += f"\n  Style: {style_desc}"
                result += f"\n  Brightness: {brightness:.0f}"
                result += f"\n  Sharpness: {sharpness:.0f}"
                result += f"\n  Global Contrast: {global_contrast:.1f}"
                result += f"\n  Local Contrast: {local_contrast:.1f}"
            
            return result
            
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Convenience functions
def check_image_quality(image_path: str, show_details: bool = False) -> str:
    """
    Enhanced image quality checker for wedding photography
    Now handles artistic/dramatic lighting as well as standard photography
    
    Args:
        image_path: Path to image file
        show_details: Show technical metrics and style detection
        
    Returns:
        Quality assessment with style-aware evaluation
    """
    checker = WeddingImageQualityChecker()
    return checker.check_quality(image_path, show_details)

def check_multiple_images(image_paths: List[str], show_details: bool = False) -> None:
    """Check multiple images and print results with style detection"""
    checker = WeddingImageQualityChecker()
    
    print("=== WEDDING PHOTOGRAPHY QUALITY ASSESSMENT ===")
    for i, path in enumerate(image_paths, 1):
        result = checker.check_quality(path, show_details)
        print(f"\nImage {i}: {result}")

def batch_wedding_assessment(folder_path: str) -> Dict[str, int]:
    """
    Assess all images in a folder and return summary statistics
    
    Args:
        folder_path: Path to folder containing images
        
    Returns:
        Dictionary with counts of each quality rating
    """
    import os
    
    checker = WeddingImageQualityChecker()
    results = {"EXCELLENT": 0, "GOOD": 0, "ACCEPTABLE": 0, "POOR": 0, "FAIL": 0}
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            full_path = os.path.join(folder_path, filename)
            assessment = checker.check_quality(full_path)
            
            # Extract verdict from assessment string
            for verdict in results.keys():
                if verdict in assessment:
                    results[verdict] += 1
                    break
    
    return results

# ==== USAGE EXAMPLES ====
if __name__ == "__main__":
    # Single image check with enhanced assessment
    image_file = r"C:\Users\HP\OneDrive\Desktop\photo\DSC07966.JPG"
    
    print("=== ENHANCED PHOTOGRAPHY ASSESSMENT ===")
    result = check_image_quality(image_file, show_details=True)
    print(result)
 