import cv2
import numpy as np

def check_image_quality(image_filename):
    """
    Quick image quality check - just the result and main reason.
    
    Args:
        image_filename: Name of the uploaded image file
        
    Returns:
        Simple string with quality verdict and main issue
    """
    try:
        img = cv2.imread(image_filename)
        
        if img is None:
            return f"❌ Error: Could not read '{image_filename}'"
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        h, s, v = cv2.split(hsv)
        l_channel = cv2.split(lab)[0]
        
        # Brightness check
        brightness = 0.7 * np.mean(l_channel) + 0.3 * np.mean(v)
        v_std = np.std(v)
        
        if v_std > 60:
            bright_thresholds = [35, 55, 200, 220]
        else:
            bright_thresholds = [45, 65, 190, 210]
        
        if brightness <= bright_thresholds[0]:
            return "❌ FAIL - Too dark (severely underexposed)"
        elif brightness <= bright_thresholds[1]:
            return "⚠️ POOR - Too dark (underexposed)"
        elif brightness >= bright_thresholds[3]:
            return "❌ FAIL - Too bright (severely overexposed)"
        elif brightness >= bright_thresholds[2]:
            return "⚠️ POOR - Too bright (overexposed)"
        
        # Sharpness check
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        height, width = img.shape[:2]
        size_factor = min(1.0, (width * height) / (1920 * 1080))
        sharp_threshold = 25 * size_factor
        
        if sharpness < sharp_threshold * 0.3:
            return "❌ FAIL - Too blurry"
        
        # Saturation check
        saturation = np.mean(s)
        if saturation < 20:
            return "⚠️ POOR - Colors too washed out"
        elif saturation > 240:
            return "⚠️ POOR - Colors oversaturated"
        
        # Contrast check
        contrast = np.std(v)
        if contrast < 25:
            return "⚠️ POOR - Very low contrast"
        
        # Exposure clipping check
        total_pixels = height * width
        overexposed = np.sum(v > 250) / total_pixels
        underexposed = np.sum(v < 15) / total_pixels
        
        if overexposed > 0.15:
            return "⚠️ POOR - Too many blown highlights"
        elif underexposed > 0.2:
            return "⚠️ POOR - Too many crushed shadows"
        
        # If we get here, image passed all critical checks
        if sharpness > sharp_threshold * 1.5 and contrast > 65:
            return "✅ EXCELLENT - Great quality image"
        elif sharpness > sharp_threshold and contrast > 45:
            return "✅ GOOD - Image quality is fine"
        else:
            return "✅ OK - Acceptable quality"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==== USAGE ====
if __name__ == "__main__":
    # Replace with your uploaded filename
    image_file = r"C:\Users\HP\OneDrive\Desktop\photo\ARP_8178.JPG"
    
    result = check_image_quality(image_file)
    print(result)