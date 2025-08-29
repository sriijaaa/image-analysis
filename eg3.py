import cv2
import numpy as np
from typing import List, Tuple

class ImageQualityChecker:
    def __init__(self, 
                 brightness_range: Tuple[int, int] = (40, 220),
                 sharpness_threshold: float = 100,
                 contrast_threshold: float = 25):
        """Simple Image Quality Assessment"""
        self.brightness_range = brightness_range
        self.sharpness_threshold = sharpness_threshold
        self.contrast_threshold = contrast_threshold
        
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)
    
    def _detect_issues(self, img: np.ndarray) -> Tuple[str, List[str]]:
        """Detect quality issues and return verdict with explanations"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)
        
        issues = []
        
        # Basic measurements
        brightness = np.mean(Y)
        sharpness = self._calculate_sharpness(gray)
        contrast = np.std(Y)
        saturation = (np.var(Cr) + np.var(Cb)) / 2
        
        total_pixels = img.shape[0] * img.shape[1]
        overexposed = np.sum(Y > 250) / total_pixels
        underexposed = np.sum(Y < 10) / total_pixels
        
        # Critical failures
        if brightness < self.brightness_range[0]:
            return "FAIL", ["Too dark - image is underexposed, details are lost in shadows"]
        elif brightness > self.brightness_range[1]:
            return "FAIL", ["Too bright - image is overexposed, highlights are blown out"]
        elif sharpness < self.sharpness_threshold * 0.3:
            return "FAIL", ["Too blurry - image lacks sharp focus throughout"]
        
        # Quality issues
        if saturation < 200:
            issues.append("Colors look washed out and dull")
        elif saturation > 5000:
            issues.append("Colors are oversaturated and unnatural")
        
        if contrast < self.contrast_threshold:
            issues.append("Low contrast makes the image appear flat")
        
        if overexposed > 0.15:
            issues.append("Bright areas are overexposed and lost detail")
        elif underexposed > 0.2:
            issues.append("Dark areas are underexposed and lost detail")
        
        # Determine verdict
        if len(issues) == 0:
            if sharpness > self.sharpness_threshold * 2 and contrast > 70:
                return "EXCELLENT", ["Sharp focus, well-balanced exposure, and natural colors"]
            elif sharpness > self.sharpness_threshold and contrast > 40:
                return "GOOD", ["Well-exposed with good sharpness and color balance"]
            else:
                return "ACCEPTABLE", ["Meets basic technical requirements"]
        elif len(issues) <= 2:
            return "ACCEPTABLE", issues
        else:
            return "POOR", issues
    
    def check_quality(self, image_path: str, show_details: bool = False) -> str:
        """
        Check image quality and explain the result
        
        Args:
            image_path: Path to the image file
            show_details: Whether to show technical metrics
            
        Returns:
            Quality assessment string
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return f"❌ Error: Could not read '{image_path}'"
            
            verdict, explanations = self._detect_issues(img)
            
            # Format output
            emoji_map = {
                'EXCELLENT': '✅',
                'GOOD': '✅', 
                'ACCEPTABLE': '✅',
                'POOR': '⚠️',
                'FAIL': '❌'
            }
            
            result = f"{emoji_map[verdict]} {verdict} - {explanations[0]}"
            
            if len(explanations) > 1:
                result += "\nAdditional issues:"
                for issue in explanations[1:]:
                    result += f"\n  • {issue}"
            
            if show_details:
                # Add key metrics
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                Y = ycrcb[:,:,0]
                
                brightness = np.mean(Y)
                sharpness = self._calculate_sharpness(gray)
                contrast = np.std(Y)
                
                result += f"\nTechnical details: Brightness: {brightness:.0f}, Sharpness: {sharpness:.0f}, Contrast: {contrast:.1f}"
            
            return result
            
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Simple function interface (maintains your original style)
def check_image_quality(image_path: str, show_details: bool = False) -> str:
    """
    Simple image quality checker - improved version of your original function
    
    Args:
        image_path: Path to image file
        show_details: Show technical metrics
        
    Returns:
        Quality assessment with explanation
    """
    checker = ImageQualityChecker()
    return checker.check_quality(image_path, show_details)

def check_multiple_images(image_paths: List[str]) -> None:
    """Check multiple images and print results"""
    checker = ImageQualityChecker()
    
    for path in image_paths:
        result = checker.check_quality(path)
        print(f"\n{result}")

# ==== USAGE ====
if __name__ == "__main__":
    # Single image check
    image_file = r"C:\Users\HP\OneDrive\Desktop\photo\DSC08382.JPG"
    
    print("=== IMAGE QUALITY ASSESSMENT ===")
    result = check_image_quality(image_file, show_details=True)
    print(result)
