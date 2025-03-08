import numpy as np
import cv2
from skimage import filters, feature, segmentation, measure
from scipy import ndimage
import matplotlib.pyplot as plt
import re
import pytesseract

def preprocess_image(image, sigma=1.0):
    """Apply preprocessing steps to enhance image for analysis."""
    # Normalize image to 0-1 range
    if image.max() > 0:
        normalized = (image - image.min()) / (image.max() - image.min())
    else:
        normalized = image
    
    # Apply Gaussian smoothing to reduce noise
    smoothed = filters.gaussian(normalized, sigma=sigma)
    return smoothed

def detect_edges(image, sigma=1.0, method='sobel'):
    """Detect edges in the image using various methods."""
    if method == 'sobel':
        edges = filters.sobel(image)
    elif method == 'canny':
        edges = feature.canny(image, sigma=sigma)
    elif method == 'scharr':
        edges = filters.scharr(image)
    else:
        raise ValueError(f"Unsupported edge detection method: {method}")
    
    return edges

def auto_threshold(image, method='otsu'):
    """Apply automatic thresholding to segment the image."""
    if method == 'otsu':
        threshold = filters.threshold_otsu(image)
    elif method == 'yen':
        threshold = filters.threshold_yen(image)
    elif method == 'mean':
        threshold = filters.threshold_mean(image)
    else:
        raise ValueError(f"Unsupported thresholding method: {method}")
        
    binary = image > threshold
    return binary, threshold

def find_interfaces(binary_image, min_size=100):
    """Find interfaces between regions in a binary image."""
    # Label connected regions
    labeled = measure.label(binary_image)
    
    # Get region properties
    regions = measure.regionprops(labeled)
    
    # Filter regions by size
    valid_regions = [region for region in regions if region.area >= min_size]
    
    # Get boundaries
    boundaries = segmentation.find_boundaries(labeled)
    
    return labeled, valid_regions, boundaries

def calibrate_scale(image, known_distance=None, pixel_size=None):
    """Calibrate the image scale to physical dimensions."""
    if pixel_size is not None:
        # Convert pixel_size to nm/pixel if it's not already
        return pixel_size
    elif known_distance is not None:
        # Here we would implement calibration using a known feature
        # For now, return a default value
        return 1.0
    else:
        # Default value if no calibration info provided
        return 1.0
        
def plot_results(image, edges=None, boundaries=None, measurements=None, figsize=(12, 8)):
    """Visualize analysis results."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Edge detection
    if edges is not None:
        axes[1].imshow(edges, cmap='magma')
        axes[1].set_title('Edge Detection')
        axes[1].axis('off')
    
    # Segmentation results
    if boundaries is not None:
        overlay = np.copy(image)
        if overlay.ndim == 2:
            overlay = np.stack([overlay]*3, axis=-1)
        overlay[boundaries, 0] = 1.0  # Mark boundaries in red
        overlay[boundaries, 1:] = 0.0
        
        axes[2].imshow(overlay)
        axes[2].set_title('Detected Interfaces')
        
        # Add thickness measurements if available
        if measurements is not None:
            for pos, thickness in measurements:
                y, x = pos
                axes[2].plot(x, y, 'go', markersize=8)
                axes[2].text(x+5, y, f"{thickness:.2f} nm", color='white', 
                             backgroundcolor='black', fontsize=8)
        
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def extract_image_metadata(image):
    """Extract metadata from the bottom part of a TEM/SEM image.
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary containing extracted metadata
    """
    # Assume metadata is in the bottom 10% of the image
    height, width = image.shape[:2]
    metadata_region = image[int(height * 0.9):, :]
    
    # Convert to proper format for OCR if needed
    if len(metadata_region.shape) == 2:  # Grayscale
        metadata_region = (metadata_region * 255).astype(np.uint8)
    
    # Use OCR to extract text
    try:
        text = pytesseract.image_to_string(metadata_region)
        
        # Parse common metadata fields
        metadata = {}
        
        # Extract magnification
        mag_match = re.search(r'Mag\s*=\s*(\d+\.?\d*)\s*[kK]?[xX]', text)
        if mag_match:
            mag_value = float(mag_match.group(1))
            if 'k' in text[mag_match.end()-2:mag_match.end()].lower():
                mag_value *= 1000
            metadata['magnification'] = mag_value
        
        # Extract working distance
        wd_match = re.search(r'WD\s*=\s*(\d+\.?\d*)\s*mm', text)
        if wd_match:
            metadata['working_distance'] = float(wd_match.group(1))
        
        # Extract EHT/voltage
        eht_match = re.search(r'EHT\s*=\s*(\d+\.?\d*)\s*kV', text)
        if eht_match:
            metadata['voltage'] = float(eht_match.group(1))
        
        # Extract date if present
        date_match = re.search(r'Date:?\s*(\d+\s*\w+\s*\d{4})', text)
        if date_match:
            metadata['date'] = date_match.group(1)
            
        # Extract signal type
        signal_match = re.search(r'Signal\s*[A-Z]\s*=\s*([A-Za-z0-9]+)', text)
        if signal_match:
            metadata['signal'] = signal_match.group(1)
            
        return metadata
    except ImportError:
        print("Warning: pytesseract not installed. Cannot extract metadata text.")
        return {}
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {}

def separate_image_and_analysis(image):
    """Separate the original image from analysis overlays and metadata regions.
    
    Args:
        image: Input image that may contain analysis overlays and metadata
        
    Returns:
        Tuple of (original_image, analysis_image, metadata_region)
    """
    if len(image.shape) < 2:
        return image, None, None
        
    height, width = image.shape[:2]
    
    # Check if image has two parts side by side (original and analysis)
    if width > height * 1.8:  # Likely has side-by-side panels
        midpoint = width // 2
        original_image = image[:, :midpoint]
        analysis_image = image[:, midpoint:]
        
        # Extract metadata region (bottom 10% of original image)
        metadata_height = int(height * 0.1)
        metadata_region = original_image[height - metadata_height:, :]
        
        # Remove metadata region from original image
        original_image = original_image[:height - metadata_height, :]
        
        return original_image, analysis_image, metadata_region
    else:
        # Just extract metadata from bottom
        metadata_height = int(height * 0.1)
        metadata_region = image[height - metadata_height:, :]
        original_image = image[:height - metadata_height, :]
        
        return original_image, None, metadata_region