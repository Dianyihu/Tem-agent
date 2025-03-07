import numpy as np
import cv2
from skimage import filters, feature, segmentation, measure
from scipy import ndimage
import matplotlib.pyplot as plt

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