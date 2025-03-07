import numpy as np
from scipy import ndimage
from skimage import measure, morphology, feature

def measure_film_thickness(labeled_image, scale_factor=1.0):
    """Measure thin film thickness from a labeled image."""
    # Assume film is above substrate (lower region numbers are typically higher in the image)
    unique_labels = np.unique(labeled_image)
    
    if len(unique_labels) < 3:  # Need at least background, film, and substrate
        return [], 0, 0
    
    measurements = []
    thicknesses = []
    
    # Get regions and their properties
    regions = measure.regionprops(labeled_image)
    
    # Sort regions by y-coordinate (top to bottom)
    sorted_regions = sorted(regions, key=lambda r: r.centroid[0])
    
    # Identify film and substrate (assuming film is above substrate)
    if len(sorted_regions) >= 2:
        film_region = sorted_regions[0]  # Topmost significant region
        substrate_region = sorted_regions[1]  # Next region
        
        film_label = film_region.label
        substrate_label = substrate_region.label
        
        # Find the boundary between film and substrate
        height, width = labeled_image.shape
        interface_points = []
        
        # Scan each column to find the interface
        for col in range(width):
            # Extract the column
            column = labeled_image[:, col]
            
            # Find where values change from film to substrate
            transitions = np.where(np.diff(column) != 0)[0]
            
            for t in transitions:
                if (column[t] == film_label and column[t+1] == substrate_label) or \
                   (column[t] == substrate_label and column[t+1] == film_label):
                    interface_points.append((t, col))
        
        if interface_points:
            # Calculate film thickness for each column
            film_top_points = []
            
            for col in range(width):
                column = labeled_image[:, col]
                film_indices = np.where(column == film_label)[0]
                
                if len(film_indices) > 0:
                    film_top = film_indices[0]  # Top edge of film
                    film_top_points.append((film_top, col))
            
            # Match interface points with top points to calculate thickness
            for (interface_y, col) in interface_points:
                # Find corresponding top point for this column
                top_points = [p for p in film_top_points if p[1] == col]
                
                if top_points:
                    film_top = top_points[0][0]
                    thickness = abs(interface_y - film_top) * scale_factor
                    measurements.append(((interface_y, col), thickness))
                    thicknesses.append(thickness)
    
    # Calculate statistics
    if thicknesses:
        mean_thickness = np.mean(thicknesses)
        std_thickness = np.std(thicknesses)
    else:
        mean_thickness = 0
        std_thickness = 0
    
    return measurements, mean_thickness, std_thickness

def measure_film_profile(labeled_image, film_label, substrate_label, scale_factor=1.0):
    """Measure film thickness profile across the image width."""
    height, width = labeled_image.shape
    profile = np.zeros(width)
    
    for col in range(width):
        column = labeled_image[:, col]
        
        # Find film-substrate interface
        transitions = np.where(np.diff(column) != 0)[0]
        interface_point = None
        
        for t in transitions:
            if (column[t] == film_label and column[t+1] == substrate_label) or \
               (column[t] == substrate_label and column[t+1] == film_label):
                interface_point = t
                break
        
        if interface_point is not None:
            # Find top of film
            film_indices = np.where(column == film_label)[0]
            if len(film_indices) > 0:
                film_top = film_indices[0]
                thickness = abs(interface_point - film_top) * scale_factor
                profile[col] = thickness
    
    return profile

def detect_layers(image, min_size=100, max_regions=5):
    """Detect multiple layers in a TEM image."""
    # Apply gradient-based edge detection
    gradient = feature.canny(image, sigma=2.0)
    
    # Close gaps in the edges
    closed_gradient = morphology.closing(gradient, morphology.square(3))
    
    # Label the regions
    labeled = measure.label(~closed_gradient)
    
    # Get region properties
    regions = measure.regionprops(labeled)
    
    # Filter regions by size and keep only the largest ones
    valid_regions = [r for r in regions if r.area >= min_size]
    valid_regions = sorted(valid_regions, key=lambda r: r.area, reverse=True)[:max_regions]
    
    # Create a new labeled image with only the valid regions
    result_labeled = np.zeros_like(labeled)
    for i, region in enumerate(valid_regions, 1):
        mask = labeled == region.label
        result_labeled[mask] = i
    
    return result_labeled, valid_regions 