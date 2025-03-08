#!/usr/bin/env python3
"""FinFET structure analyzer for TEM images."""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, measure, morphology, filters, color
from skimage.feature import canny as canny_filter
from scipy.ndimage import gaussian_filter

from tem_agent.core import BaseAnalyzer, AnalysisResult, ImageHandler, Metrics

class FinfetResult(AnalysisResult):
    """Class for FinFET analysis results."""
    
    def _get_text_summary(self):
        """Generate a text summary of FinFET analysis results."""
        if not self.data or not self.metadata:
            return "No FinFET measurements available."
            
        summary = f"FinFET Structure Analysis Results\n"
        summary += f"===============================\n\n"
        
        # Add image info
        if 'image_info' in self.metadata:
            info = self.metadata['image_info']
            summary += f"Image: {info.get('filename', 'Unknown')}\n"
            summary += f"Dimensions: {info.get('shape', 'Unknown')}\n"
            summary += f"Scale: {info.get('scale_factor', 'Unknown')}\n\n"
        
        # Add fin width statistics
        if 'width_statistics' in self.metadata:
            stats = self.metadata['width_statistics']
            summary += f"Fin Width Statistics:\n"
            summary += f"  Mean width: {stats.get('mean', 0):.2f} nm\n"
            summary += f"  Standard deviation: {stats.get('std', 0):.2f} nm\n"
            summary += f"  Min width: {stats.get('min', 0):.2f} nm\n"
            summary += f"  Max width: {stats.get('max', 0):.2f} nm\n"
            summary += f"  Number of fins: {stats.get('count', 0)}\n\n"
            
        # Add fin pitch statistics
        if 'pitch_statistics' in self.metadata:
            stats = self.metadata['pitch_statistics']
            summary += f"Fin Pitch Statistics:\n"
            summary += f"  Mean pitch: {stats.get('mean', 0):.2f} nm\n"
            summary += f"  Standard deviation: {stats.get('std', 0):.2f} nm\n"
            summary += f"  Number of measurements: {stats.get('count', 0)}\n\n"
            
        # Add fin height statistics
        if 'height_statistics' in self.metadata:
            stats = self.metadata['height_statistics']
            summary += f"Fin Height Statistics:\n"
            summary += f"  Mean height: {stats.get('mean', 0):.2f} nm\n"
            summary += f"  Standard deviation: {stats.get('std', 0):.2f} nm\n"
            summary += f"  Number of measurements: {stats.get('count', 0)}\n\n"
            
        # Add coating statistics if available
        if 'coating_statistics' in self.metadata:
            stats = self.metadata['coating_statistics']
            if stats.get('count', 0) > 0:
                summary += f"Coating Thickness Statistics:\n"
                summary += f"  Mean thickness: {stats.get('mean', 0):.2f} nm\n"
                summary += f"  Standard deviation: {stats.get('std', 0):.2f} nm\n"
                summary += f"  Min thickness: {stats.get('min', 0):.2f} nm\n"
                summary += f"  Max thickness: {stats.get('max', 0):.2f} nm\n\n"
            
        # Add individual measurements
        summary += "Detailed Fin Measurements:\n"
        for i, m in enumerate(self.data.get('fin_widths', [])):
            width = m.get('width', 0) * self.metadata.get('scale_factor', 1.0)
            summary += f"  Fin {i+1}: Width {width:.2f} nm\n"
            
        # Add fin pitch measurements
        if 'fin_pitches' in self.data and self.data['fin_pitches']:
            summary += "\nFin Pitch Measurements:\n"
            for i, pitch in enumerate(self.data['fin_pitches']):
                pitch_value = pitch.get('pitch', 0) * self.metadata.get('scale_factor', 1.0)
                summary += f"  Pitch {i+1}: {pitch_value:.2f} nm\n"
                
        # Add fin height measurements
        if 'fin_heights' in self.data and self.data['fin_heights']:
            summary += "\nFin Height Measurements:\n"
            for i, height in enumerate(self.data['fin_heights']):
                height_value = height.get('height', 0) * self.metadata.get('scale_factor', 1.0)
                summary += f"  Height {i+1}: {height_value:.2f} nm\n"
            
        # Add coating measurements if available
        if 'coating_measurements' in self.data and self.data['coating_measurements']:
            summary += "\nDetailed Coating Measurements:\n"
            for i, coating in enumerate(self.data['coating_measurements']):
                avg = coating.get('avg_thickness', 0)
                std = coating.get('std_thickness', 0)
                summary += f"  Fin {i+1} coating: {avg:.2f} ± {std:.2f} nm\n"
            
        return summary


class FinfetAnalyzer(BaseAnalyzer):
    """Analyzer for measuring FinFET structures in TEM images."""
    
    def __init__(self, pixel_size=None, u_shaped=False, detect_dark_fins=True):
        """Initialize FinFET analyzer.
        
        Args:
            pixel_size: Physical size per pixel in nm/pixel, if known
            u_shaped: Set to True if analyzing U-shaped FinFET structures
            detect_dark_fins: Set to True to detect dark regions as fins (default behavior for FinFET)
        """
        super().__init__(pixel_size=pixel_size)
        self.u_shaped = u_shaped
        self.detect_dark_fins = detect_dark_fins
        self.fin_mask = None
        self.fin_regions = None
        self.fin_widths = None
        self.fin_pitches = None
        self.fin_heights = None
        self.coating_edges = None
        self.coating_measurements = None
        self.width_statistics = None
        self.pitch_statistics = None
        self.height_statistics = None
        self.coating_statistics = None
        
    def load_image(self, file_path):
        """Load an image file containing FinFET structures."""
        self.image, self.metadata, scale_factors = ImageHandler.load_image(file_path)
        self.filename = file_path
        
        # Set scale factor if available from metadata
        if scale_factors is not None:
            self.scale_factor = scale_factors[0]  # Use x-scale as default
            print(f"Loaded image with scale factor: {self.scale_factor} nm/pixel")
        elif self.pixel_size is not None:
            self.scale_factor = self.pixel_size
            print(f"Using provided pixel size: {self.scale_factor} nm/pixel")
            
        return self.image
    
    def preprocess(self, sigma=2.0, denoise=True, normalize=True, **kwargs):
        """Preprocess the loaded image."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        self.processed_image = ImageHandler.preprocess(
            self.image, 
            sigma=sigma, 
            denoise=denoise, 
            normalize=normalize,
            **kwargs
        )
        
        return self.processed_image
    
    def detect_fins(self, sigma=2.0, thresh_method='otsu', min_size=100, max_regions=10):
        """Detect FinFET structures in the image with smooth boundary lines.
        
        Args:
            sigma: Sigma for Gaussian smoothing
            thresh_method: Thresholding method ('otsu', 'yen', 'mean')
            min_size: Minimum size of regions to keep
            max_regions: Maximum number of regions to keep
            
        This implementation focuses on creating smooth contour lines with
        minimal angle deviations at the boundary between fins and coating.
        """
        if not hasattr(self, 'processed_image') or self.processed_image is None:
            self.preprocess(sigma=sigma)
        
        # Apply thresholding to segment the image
        from skimage.filters import threshold_otsu, threshold_yen, threshold_mean
        from skimage import measure
        
        if thresh_method == 'otsu':
            thresh = threshold_otsu(self.processed_image)
        elif thresh_method == 'yen':
            thresh = threshold_yen(self.processed_image)
        elif thresh_method == 'mean':
            thresh = threshold_mean(self.processed_image)
        else:
            raise ValueError(f"Unsupported threshold method: {thresh_method}")
        
        # SMOOTH CONTOUR LINE APPROACH: 
        # 1. Use strong Gaussian blur to create smooth intensity gradients
        # 2. Apply simple thresholding on the blurred image for smooth boundaries
        # 3. Use large morphological operations to ensure smooth contours with minimal angle changes
        
        # Set detect_dark_fins to True
        self.detect_dark_fins = True
        height, width = self.processed_image.shape
        
        # Step 1: Apply strong Gaussian blur to smooth out noise
        blurred_image = gaussian_filter(self.processed_image, sigma=4.0)
        
        # Step 2: Simple thresholding for coating detection
        # Using a moderately high threshold to detect the bright coating
        bright_thresh = thresh * 1.15
        bright_regions = blurred_image > bright_thresh
        
        # Apply strong morphological operations for smooth boundaries
        coating_mask = morphology.binary_closing(bright_regions, morphology.disk(8))
        coating_mask = morphology.remove_small_objects(coating_mask, min_size=min_size)
        coating_mask = morphology.remove_small_holes(coating_mask, area_threshold=100)
        
        # Store the coating mask
        self.bright_coating = coating_mask
        
        # Step 3: Create a mask for everything below the coating
        below_coating_mask = np.zeros_like(self.processed_image, dtype=bool)
        
        for col in range(width):
            coating_pixels = np.where(coating_mask[:, col])[0]
            if len(coating_pixels) > 0:
                bottom_row = coating_pixels.max()
                # Start several pixels below to ensure clear separation
                if bottom_row + 4 < height:  # Increased from 2 to 4 for clearer separation
                    below_coating_mask[bottom_row + 4:, col] = True
        
        # Step 4: Apply strong smoothing to define fins
        # First, apply very large closing operation for smooth boundaries
        fin_mask = morphology.binary_closing(below_coating_mask, morphology.disk(10))
        
        # Slight erosion and dilation to further smooth the boundary
        fin_mask = morphology.binary_erosion(fin_mask, morphology.disk(2))
        fin_mask = morphology.binary_dilation(fin_mask, morphology.disk(2))
        
        # Step 5: Split into left and right fins
        middle = width // 2
        
        left_half = np.zeros_like(self.processed_image, dtype=bool)
        right_half = np.zeros_like(self.processed_image, dtype=bool)
        
        left_half[:, :middle] = True
        right_half[:, middle:] = True
        
        left_fin = fin_mask & left_half
        right_fin = fin_mask & right_half
        
        # Step 6: Ensure connection to the bottom of the image
        bottom_portion = int(height * 0.8)
        bottom_mask = np.zeros_like(fin_mask, dtype=bool)
        bottom_mask[bottom_portion:, :] = True
        
        # Initialize final fins mask and labeled image
        final_fins = np.zeros_like(self.processed_image, dtype=bool)
        labeled_fins = np.zeros_like(self.processed_image, dtype=np.int32)
        
        # Find regions connected to the bottom
        left_labeled, _ = measure.label(left_fin, return_num=True)
        
        left_regions = []
        for region_id in range(1, np.max(left_labeled) + 1):
            region_mask = left_labeled == region_id
            if np.any(region_mask & bottom_mask):
                left_regions.append((region_id, np.sum(region_mask)))
        
        # Add the largest connected region in the left half
        if left_regions:
            left_regions.sort(key=lambda x: x[1], reverse=True)
            largest_left_id = left_regions[0][0]
            left_fin_mask = left_labeled == largest_left_id
            
            # Add to final fins
            final_fins |= left_fin_mask
            labeled_fins[left_fin_mask] = 1  # Label as fin 1
        
        # Do the same for the right fin
        right_labeled, _ = measure.label(right_fin, return_num=True)
        
        right_regions = []
        for region_id in range(1, np.max(right_labeled) + 1):
            region_mask = right_labeled == region_id
            if np.any(region_mask & bottom_mask):
                right_regions.append((region_id, np.sum(region_mask)))
        
        # Add the largest connected region in the right half
        if right_regions:
            right_regions.sort(key=lambda x: x[1], reverse=True)
            largest_right_id = right_regions[0][0]
            right_fin_mask = right_labeled == largest_right_id
            
            # Add to final fins
            final_fins |= right_fin_mask
            labeled_fins[right_fin_mask] = 2  # Label as fin 2
        
        # Step 7: Final touch - ensure the boundary is extremely smooth
        # Apply one final closing with a very large disk
        if np.any(final_fins):
            final_fins = morphology.binary_closing(final_fins, morphology.disk(12))
            
            # Re-separate into left and right fins
            left_fin_final = final_fins & left_half
            right_fin_final = final_fins & right_half
            
            # Re-label
            labeled_fins = np.zeros_like(final_fins, dtype=np.int32)
            if np.any(left_fin_final):
                labeled_fins[left_fin_final] = 1
            if np.any(right_fin_final):
                labeled_fins[right_fin_final] = 2
        
        # Print detection message
        print(f"Detecting fins with extremely smooth boundary lines")
        
        # Store the fin mask and labeled image
        self.fin_mask = final_fins
        self.labeled_fins = labeled_fins
        
        # Get region properties
        self.fin_regions = Metrics.measure_region_properties(
            labeled_fins, 
            intensity_image=self.processed_image,
            properties=['area', 'centroid', 'bbox', 'orientation', 'major_axis_length', 'minor_axis_length']
        )
        
        print(f"Detected {len(self.fin_regions)} fin structures")
        
        return self.fin_regions
    
    def measure_fins(self, num_profiles=10):
        """Measure the width of detected fins."""
        if not hasattr(self, 'fin_regions') or not self.fin_regions:
            self.detect_fins()
            
        fin_widths = []
        
        for i, region in enumerate(self.fin_regions):
            bbox = region.get('bbox', None)
            if bbox is None:
                continue
                
            # Get region bounding box
            min_row, min_col, max_row, max_col = bbox
            
            # For U-shaped fins, use a different measurement approach
            if self.u_shaped:
                # For U-shaped fins, measure the distance between outer edges
                width_measurements = []
                
                # Take multiple profiles across the fin
                for j in range(num_profiles):
                    # Determine position for this profile (evenly distributed)
                    y_pos = min_row + int((max_row - min_row) * j / (num_profiles - 1))
                    
                    # Extract horizontal profile at this position
                    profile = self.labeled_fins[y_pos, min_col:max_col+1] > 0
                    
                    # Find edges in this profile
                    if np.any(profile):
                        left_edge = np.where(profile)[0][0]
                        right_edge = np.where(profile)[0][-1]
                        width = right_edge - left_edge + 1
                        width_measurements.append({
                            'y_position': y_pos,
                            'width': width,
                            'left_x': min_col + left_edge,
                            'right_x': min_col + right_edge
                        })
                
                # Calculate average width
                if width_measurements:
                    widths = [m['width'] for m in width_measurements]
                    avg_width = np.mean(widths)
                    fin_widths.append({
                        'region_index': i,
                        'width': avg_width,
                        'width_nm': avg_width * self.scale_factor,
                        'profiles': width_measurements
                    })
            else:
                # For standard fins, use minor axis length as a proxy for width
                minor_axis = region.get('minor_axis_length', 0)
                fin_widths.append({
                    'region_index': i,
                    'width': minor_axis,
                    'width_nm': minor_axis * self.scale_factor,
                    'centroid': region.get('centroid', (0, 0))
                })
                
        self.fin_widths = fin_widths
        
        # Calculate statistics
        widths = [fw['width'] for fw in fin_widths]
        if widths:
            self.width_statistics = {
                'count': len(widths),
                'mean': np.mean(widths) * self.scale_factor,
                'std': np.std(widths) * self.scale_factor,
                'min': np.min(widths) * self.scale_factor,
                'max': np.max(widths) * self.scale_factor
            }
        else:
            self.width_statistics = {'count': 0}
            
        return self.fin_widths
    
    def detect_coating(self, sigma=1.0, edge_method='canny'):
        """Detect coating on fin structures with smooth boundary lines."""
        if not hasattr(self, 'fin_mask') or self.fin_mask is None:
            self.detect_fins()
            
        # Use the bright_coating we already created during fin detection
        if hasattr(self, 'bright_coating') and self.bright_coating is not None:
            # Get a clean version of the coating mask that doesn't overlap with fins
            coating_mask = self.bright_coating.copy()
            coating_mask = coating_mask & ~self.fin_mask
            
            # Apply a large disk for final smoothing to ensure smooth contours
            coating_mask = morphology.binary_closing(coating_mask, morphology.disk(8))
            
            # Store the smooth coating mask
            self.coating_aoi = coating_mask
            
            # Create a boundary representation with minimal angle deviation
            # First get the boundary using morphological gradient
            dilated_coating = morphology.binary_dilation(coating_mask, morphology.disk(2))
            eroded_coating = morphology.binary_erosion(coating_mask, morphology.disk(2))
            boundary = dilated_coating & ~eroded_coating
            
            # Find the interface with the fins
            dilated_fins = morphology.binary_dilation(self.fin_mask, morphology.disk(2))
            interface = boundary & dilated_fins
            
            # If the interface is too small, use the general boundary
            if np.sum(interface) < 50:
                interface = boundary
            
            # Apply smoothing to the interface for minimal angle deviation
            smooth_interface = gaussian_filter(interface.astype(float), sigma=1.5) > 0.2
            
            # Store the smooth interface as our edges
            self.coating_edges = smooth_interface
            
            return smooth_interface, coating_mask
        
        # If we don't have a coating mask already, create one
        # Using a 3-step process to ensure smooth contours:
        
        # Step 1: Apply strong Gaussian blur to the image
        blurred_image = gaussian_filter(self.processed_image, sigma=3.0)
        
        # Step 2: Threshold the blurred image for smoother regions
        if not hasattr(self, 'processed_image') or self.processed_image is None:
            self.preprocess(sigma=sigma)
            
        # Get threshold value
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(self.processed_image)
        
        # Create bright regions mask (coating)
        bright_thresh = thresh * 1.15
        bright_regions = blurred_image > bright_thresh
        
        # Step 3: Apply morphological operations with large structuring elements
        # for smooth boundaries with minimal angle deviation
        coating_mask = morphology.binary_closing(bright_regions, morphology.disk(10))
        coating_mask = morphology.remove_small_objects(coating_mask, min_size=100)
        coating_mask = morphology.remove_small_holes(coating_mask, area_threshold=100)
        
        # Ensure the coating doesn't overlap with fins
        if hasattr(self, 'fin_mask') and self.fin_mask is not None:
            coating_mask = coating_mask & ~self.fin_mask
        
        # Get the boundary
        dilated = morphology.binary_dilation(coating_mask, morphology.disk(2))
        eroded = morphology.binary_erosion(coating_mask, morphology.disk(2))
        boundary = dilated & ~eroded
        
        # Smooth the boundary for minimal angle deviation
        smooth_boundary = gaussian_filter(boundary.astype(float), sigma=2.0) > 0.2
        
        # Store results
        self.coating_edges = smooth_boundary
        self.coating_aoi = coating_mask
        
        return smooth_boundary, coating_mask
    
    def measure_coating(self, num_measurements=20):
        """Measure the thickness of the coating around fins."""
        if not hasattr(self, 'fin_mask') or self.fin_mask is None:
            self.detect_fins()
            
        if not hasattr(self, 'coating_edges') or self.coating_edges is None:
            self.detect_coating()
            
        coating_measurements = []
        
        # For each fin region
        for i, region in enumerate(self.fin_regions):
            bbox = region.get('bbox', None)
            if bbox is None:
                continue
                
            # Get region bounding box
            min_row, min_col, max_row, max_col = bbox
            
            # Create a mask for just this fin
            fin_mask = self.labeled_fins == (i + 1)
            
            # Dilate to get potential coating area
            dilated = morphology.binary_dilation(fin_mask, morphology.disk(5))
            coating_area = dilated & ~fin_mask
            
            # Isolate edges in the coating area
            coating_edges = self.coating_edges * coating_area
            
            # Measure distance from fin boundary to outer edge
            thickness_values = []
            
            # Get the boundary pixels of the fin
            fin_boundary = morphology.binary_dilation(fin_mask) & ~fin_mask
            
            # Get coordinates of boundary pixels
            boundary_coords = np.where(fin_boundary)
            
            # If we have too many boundary pixels, sample a subset
            if len(boundary_coords[0]) > num_measurements:
                indices = np.linspace(0, len(boundary_coords[0])-1, num_measurements, dtype=int)
                boundary_coords = (boundary_coords[0][indices], boundary_coords[1][indices])
                
            # For each boundary pixel
            for y, x in zip(*boundary_coords):
                # Find the nearest edge pixel
                # Simple approach: Look outward in radial lines
                max_dist = 20  # Maximum search distance
                found_edge = False
                
                for d in range(1, max_dist+1):
                    # Check points in a circle of radius d
                    for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                        dx = int(np.round(d * np.cos(angle)))
                        dy = int(np.round(d * np.sin(angle)))
                        
                        # Check if this point is within image bounds
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < coating_edges.shape[0] and 0 <= nx < coating_edges.shape[1]:
                            # Check if this is an edge pixel
                            if coating_edges[ny, nx]:
                                thickness_values.append(d)
                                found_edge = True
                                break
                    
                    if found_edge:
                        break
            
            # Calculate statistics for this fin's coating
            if thickness_values:
                avg_thickness = np.mean(thickness_values) * self.scale_factor
                std_thickness = np.std(thickness_values) * self.scale_factor
            else:
                avg_thickness = 0
                std_thickness = 0
                
            coating_measurements.append({
                'fin_index': i,
                'avg_thickness': avg_thickness,
                'std_thickness': std_thickness,
                'measurements': len(thickness_values)
            })
            
        self.coating_measurements = coating_measurements
        
        # Calculate overall statistics
        avg_thicknesses = [m['avg_thickness'] for m in coating_measurements if m['measurements'] > 0]
        if avg_thicknesses:
            self.coating_statistics = {
                'count': len(avg_thicknesses),
                'mean': np.mean(avg_thicknesses),
                'std': np.std(avg_thicknesses),
                'min': np.min(avg_thicknesses),
                'max': np.max(avg_thicknesses)
            }
        else:
            self.coating_statistics = {'count': 0}
            
        return self.coating_measurements
    
    def measure_fin_pitch(self):
        """Measure the pitch (spacing) between fins."""
        if not hasattr(self, 'fin_regions') or not self.fin_regions:
            self.detect_fins()
            
        # Get the centers of each fin
        fin_centers = []
        for region in self.fin_regions:
            centroid = region.get('centroid', None)
            if centroid is not None:
                fin_centers.append((centroid[1], centroid[0]))  # (x, y) format
        
        # Sort by x-coordinate to ensure we measure adjacent fins
        fin_centers.sort(key=lambda p: p[0])
        
        # Calculate pitches between adjacent fins
        pitches = []
        for i in range(len(fin_centers) - 1):
            # Calculate distance between adjacent centers
            dx = fin_centers[i+1][0] - fin_centers[i][0]
            dy = fin_centers[i+1][1] - fin_centers[i][1]
            distance = np.sqrt(dx**2 + dy**2)
            
            pitches.append({
                'fin_index_1': i,
                'fin_index_2': i+1,
                'pitch': distance,
                'pitch_nm': distance * self.scale_factor,
                'center_1': fin_centers[i],
                'center_2': fin_centers[i+1]
            })
        
        self.fin_pitches = pitches
        
        # Calculate statistics
        if pitches:
            pitch_values = [p['pitch'] for p in pitches]
            self.pitch_statistics = {
                'count': len(pitches),
                'mean': np.mean(pitch_values) * self.scale_factor,
                'std': np.std(pitch_values) * self.scale_factor if len(pitch_values) > 1 else 0,
                'min': np.min(pitch_values) * self.scale_factor,
                'max': np.max(pitch_values) * self.scale_factor
            }
        else:
            self.pitch_statistics = {'count': 0}
            
        return self.fin_pitches
        
    def measure_fin_height(self):
        """Measure the height of the fins."""
        if not hasattr(self, 'fin_regions') or not self.fin_regions:
            self.detect_fins()
            
        heights = []
        
        for i, region in enumerate(self.fin_regions):
            # Use bounding box to estimate height
            bbox = region.get('bbox', None)
            if bbox is None:
                continue
                
            # Get region bounding box
            min_row, min_col, max_row, max_col = bbox
            
            # Calculate height (vertical extent)
            height = max_row - min_row
            
            heights.append({
                'region_index': i,
                'height': height,
                'height_nm': height * self.scale_factor,
                'top_y': min_row,
                'bottom_y': max_row
            })
        
        self.fin_heights = heights
        
        # Calculate statistics
        if heights:
            height_values = [h['height'] for h in heights]
            self.height_statistics = {
                'count': len(heights),
                'mean': np.mean(height_values) * self.scale_factor,
                'std': np.std(height_values) * self.scale_factor if len(height_values) > 1 else 0,
                'min': np.min(height_values) * self.scale_factor,
                'max': np.max(height_values) * self.scale_factor
            }
        else:
            self.height_statistics = {'count': 0}
            
        return self.fin_heights
    
    def analyze(self, run_coating_analysis=True, **kwargs):
        """Analyze FinFET structures in the image.
        
        Args:
            run_coating_analysis: Whether to analyze coating thickness
            **kwargs: Additional parameters for analysis
            
        Returns:
            FinfetResult object with analysis results
        """
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        # Process parameters
        sigma = kwargs.get('sigma', 2.0)
        thresh_method = kwargs.get('thresh_method', 'otsu')
        min_size = kwargs.get('min_size', 100)
        max_regions = kwargs.get('max_regions', 10)
        
        # Detect and measure fins
        self.detect_fins(
            sigma=sigma,
            thresh_method=thresh_method,
            min_size=min_size,
            max_regions=max_regions
        )
        
        self.measure_fins(num_profiles=kwargs.get('num_profiles', 10))
        
        # Measure fin pitch and height
        self.measure_fin_pitch()
        self.measure_fin_height()
        
        # Analyze coating if requested
        coating_data = None
        self.coating_statistics = {'count': 0}  # Initialize with empty statistics
        
        if run_coating_analysis:
            try:
                coating_sigma = kwargs.get('coating_sigma', 1.0)
                edge_method = kwargs.get('edge_method', 'canny')
                
                self.detect_coating(sigma=coating_sigma, edge_method=edge_method)
                self.measure_coating(num_measurements=kwargs.get('num_coating_measurements', 20))
                coating_data = self.coating_measurements
            except Exception as e:
                print(f"Warning: Failed to analyze coating: {str(e)}")
                coating_data = None
                
        # Prepare result data
        result_data = {
            'fin_widths': self.fin_widths,
            'fin_pitches': self.fin_pitches,
            'fin_heights': self.fin_heights,
            'coating_measurements': coating_data
        }
        
        # Create result metadata
        metadata = {
            'image_info': self.get_image_info(),
            'width_statistics': self.width_statistics,
            'pitch_statistics': self.pitch_statistics,
            'height_statistics': self.height_statistics,
            'parameters': kwargs,
            'scale_factor': self.scale_factor,
            'coating_statistics': self.coating_statistics  # Always include, even if empty
        }
        
        # Create and return result object
        result = FinfetResult(
            data=result_data,
            metadata=metadata
        )
        
        self.results = result
        return result
    
    def visualize(self, show=True, figsize=(14, 10)):
        """Visualize FinFET analysis results."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        if not hasattr(self, 'fin_widths') or not self.fin_widths:
            self.analyze()
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmented fins with coating
        if hasattr(self, 'labeled_fins') and self.labeled_fins is not None:
            # Show the original image
            axes[1].imshow(self.image, cmap='gray')
            
            # Create a composite overlay showing both fins and coating
            
            # Add bright coating in gold/yellow
            if hasattr(self, 'bright_coating'):
                coating_overlay = np.zeros((*self.image.shape, 4), dtype=np.float32)
                coating_overlay[..., 0] = 1.0  # R
                coating_overlay[..., 1] = 1.0  # G
                coating_overlay[..., 2] = 0.0  # B
                coating_overlay[..., 3] = self.bright_coating * 0.5  # A
                axes[1].imshow(coating_overlay)
            
            # Add the fin overlay in blue (with alpha)
            fin_mask = self.labeled_fins > 0
            fin_overlay = np.zeros((*self.image.shape, 4), dtype=np.float32)
            fin_overlay[..., 0] = 0.0  # R
            fin_overlay[..., 1] = 0.0  # G
            fin_overlay[..., 2] = 1.0  # B
            fin_overlay[..., 3] = fin_mask * 0.5  # Add alpha channel
            axes[1].imshow(fin_overlay)
            
            axes[1].set_title('Detected Fins (Blue) & Coating (Yellow)')
        else:
            axes[1].imshow(self.image, cmap='gray')
            axes[1].set_title('No Fins Detected')
        axes[1].axis('off')
        
        # Fin measurements
        axes[2].imshow(self.image, cmap='gray')
        axes[2].set_title('Fin Measurements')
        
        # Draw fin width measurements
        if hasattr(self, 'fin_widths') and self.fin_widths:
            for fw in self.fin_widths:
                # Different visualization for U-shaped vs regular fins
                if self.u_shaped and 'profiles' in fw:
                    # For U-shaped fins, show measurement lines
                    for profile in fw['profiles']:
                        left_x = profile.get('left_x', 0)
                        right_x = profile.get('right_x', 0)
                        y_pos = profile.get('y_position', 0)
                        
                        # Draw the measurement line
                        axes[2].plot([left_x, right_x], [y_pos, y_pos], 'g-', linewidth=1)
                        
                    # Add label for this fin
                    idx = fw.get('region_index', 0)
                    width_nm = fw.get('width_nm', 0)
                    
                    # Get a representative profile for label placement
                    if fw['profiles']:
                        profile = fw['profiles'][len(fw['profiles'])//2]
                        mid_x = (profile.get('left_x', 0) + profile.get('right_x', 0)) // 2
                        y_pos = profile.get('y_position', 0)
                        
                        axes[2].text(mid_x, y_pos - 10, 
                                    f'W{idx+1}: {width_nm:.1f} nm', 
                                    color='white', fontsize=9,
                                    bbox=dict(facecolor='green', alpha=0.7),
                                    ha='center')
                else:
                    # For regular fins, use centroid and minor axis
                    idx = fw.get('region_index', 0)
                    width_nm = fw.get('width_nm', 0)
                    centroid = fw.get('centroid', (0, 0))
                    
                    if centroid != (0, 0):
                        # Draw centroid marker
                        axes[2].plot(centroid[1], centroid[0], 'go', markersize=6)
                        
                        # Add width label
                        axes[2].text(centroid[1] + 5, centroid[0] - 5, 
                                   f'W{idx+1}: {width_nm:.1f}', 
                                   color='white', fontsize=9,
                                   bbox=dict(facecolor='green', alpha=0.7))
        
        # Draw fin pitch measurements
        if hasattr(self, 'fin_pitches') and self.fin_pitches:
            for i, pitch in enumerate(self.fin_pitches):
                center1 = pitch.get('center_1', (0, 0))
                center2 = pitch.get('center_2', (0, 0))
                pitch_nm = pitch.get('pitch_nm', 0)
                
                # Draw line between centers
                axes[2].plot([center1[0], center2[0]], [center1[1], center2[1]], 'c-', linewidth=1)
                
                # Add pitch label at midpoint
                mid_x = (center1[0] + center2[0]) / 2
                mid_y = (center1[1] + center2[1]) / 2 + 15  # Offset to not overlap with fin labels
                
                axes[2].text(mid_x, mid_y, 
                           f'P{i+1}: {pitch_nm:.1f}', 
                           color='white', fontsize=9,
                           bbox=dict(facecolor='cyan', alpha=0.7),
                           ha='center')
                           
        # Draw fin height measurements
        if hasattr(self, 'fin_heights') and self.fin_heights:
            for i, height in enumerate(self.fin_heights):
                idx = height.get('region_index', 0)
                
                # Only add height labels for a few fins to avoid clutter
                if i % 2 == 0 and i < len(self.fin_regions):
                    region = self.fin_regions[idx]
                    centroid = region.get('centroid', None)
                    
                    if centroid:
                        top_y = height.get('top_y', 0)
                        bottom_y = height.get('bottom_y', 0)
                        height_nm = height.get('height_nm', 0)
                        
                        # Draw height line
                        x_pos = centroid[1] - 15  # Offset to the left of centroid
                        axes[2].plot([x_pos, x_pos], [top_y, bottom_y], 'y-', linewidth=1)
                        
                        # Add height label
                        axes[2].text(x_pos - 5, (top_y + bottom_y) / 2, 
                                   f'H{idx+1}: {height_nm:.1f}', 
                                   color='white', fontsize=9,
                                   bbox=dict(facecolor='yellow', alpha=0.7),
                                   rotation=90, ha='center', va='center')
        
        axes[2].axis('off')
        
        # Coating measurements
        axes[3].imshow(self.image, cmap='gray')
        axes[3].set_title('Coating Visualization') # Changed title to reflect no measurements
        
        # Show coating if available
        if hasattr(self, 'coating_aoi') and self.coating_aoi is not None:
            # Create mask overlay for coating area
            coating_overlay = np.zeros((*self.image.shape, 4), dtype=np.float32)
            coating_overlay[..., 0] = 1  # R
            coating_overlay[..., 1] = 0  # G
            coating_overlay[..., 2] = 0  # B
            coating_overlay[..., 3] = self.coating_aoi * 0.3  # A
            
            axes[3].imshow(coating_overlay)
            
            # Show edge detection if available
            if hasattr(self, 'coating_edges') and self.coating_edges is not None:
                # Create an edge overlay
                edge_overlay = np.zeros((*self.image.shape, 4), dtype=np.float32)
                edge_overlay[..., 0] = 1  # R
                edge_overlay[..., 1] = 0.5  # G
                edge_overlay[..., 2] = 0  # B
                edge_overlay[..., 3] = self.coating_edges * 0.7  # A
                
                axes[3].imshow(edge_overlay)
                
            # Removed the coating thickness labels as requested
        
        axes[3].axis('off')
        
        # Add overall title with statistics
        title = "FinFET Structure Analysis"
        if hasattr(self, 'width_statistics') and self.width_statistics.get('count', 0) > 0:
            fin_count = self.width_statistics.get('count', 0)
            title += f": {fin_count} fins"
            
            # Add key measurements to title
            if hasattr(self, 'width_statistics'):
                width = self.width_statistics.get('mean', 0)
                title += f", W={width:.1f} nm"
                
            if hasattr(self, 'height_statistics') and self.height_statistics.get('count', 0) > 0:
                height = self.height_statistics.get('mean', 0)
                title += f", H={height:.1f} nm"
                
            if hasattr(self, 'pitch_statistics') and self.pitch_statistics.get('count', 0) > 0:
                pitch = self.pitch_statistics.get('mean', 0)
                title += f", P={pitch:.1f} nm"
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig, axes
    
    def save_results(self, output_dir="results", plot_format='png'):
        """Save analysis results to files."""
        if self.results is None:
            raise ValueError("No results to save. Call analyze() first.")
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        
        # Save fin width measurements as CSV
        widths_file = os.path.join(output_dir, f"{base_name}_fin_widths.csv")
        with open(widths_file, 'w') as f:
            f.write("fin_id,width_px,width_nm\n")
            for fw in self.fin_widths:
                idx = fw.get('region_index', 0)
                width = fw.get('width', 0)
                width_nm = fw.get('width_nm', 0)
                f.write(f"{idx+1},{width:.4f},{width_nm:.4f}\n")
        
        # Save fin pitch measurements if available
        if hasattr(self, 'fin_pitches') and self.fin_pitches:
            pitch_file = os.path.join(output_dir, f"{base_name}_fin_pitches.csv")
            with open(pitch_file, 'w') as f:
                f.write("pitch_id,fin1_id,fin2_id,pitch_px,pitch_nm\n")
                for i, pitch in enumerate(self.fin_pitches):
                    fin1 = pitch.get('fin_index_1', 0)
                    fin2 = pitch.get('fin_index_2', 0)
                    pitch_px = pitch.get('pitch', 0)
                    pitch_nm = pitch.get('pitch_nm', 0)
                    f.write(f"{i+1},{fin1+1},{fin2+1},{pitch_px:.4f},{pitch_nm:.4f}\n")
        
        # Save fin height measurements if available
        if hasattr(self, 'fin_heights') and self.fin_heights:
            height_file = os.path.join(output_dir, f"{base_name}_fin_heights.csv")
            with open(height_file, 'w') as f:
                f.write("fin_id,height_px,height_nm\n")
                for height in self.fin_heights:
                    idx = height.get('region_index', 0)
                    height_px = height.get('height', 0)
                    height_nm = height.get('height_nm', 0)
                    f.write(f"{idx+1},{height_px:.4f},{height_nm:.4f}\n")
                
        # Save coating measurements if available
        if hasattr(self, 'coating_measurements') and self.coating_measurements:
            coating_file = os.path.join(output_dir, f"{base_name}_coating.csv")
            with open(coating_file, 'w') as f:
                f.write("fin_id,avg_thickness_nm,std_thickness_nm,num_measurements\n")
                for i, coating in enumerate(self.coating_measurements):
                    avg = coating.get('avg_thickness', 0)
                    std = coating.get('std_thickness', 0)
                    count = coating.get('measurements', 0)
                    f.write(f"{i+1},{avg:.4f},{std:.4f},{count}\n")
                    
        # Save statistics
        stats_file = os.path.join(output_dir, f"{base_name}_statistics.txt")
        with open(stats_file, 'w') as f:
            f.write(f"FinFET Analysis Statistics\n")
            f.write(f"=======================\n\n")
            
            # Fin width statistics
            if hasattr(self, 'width_statistics') and self.width_statistics:
                f.write(f"Fin Width Statistics:\n")
                f.write(f"  Number of fins: {self.width_statistics.get('count', 0)}\n")
                f.write(f"  Mean width: {self.width_statistics.get('mean', 0):.4f} nm\n")
                f.write(f"  Standard deviation: {self.width_statistics.get('std', 0):.4f} nm\n")
                f.write(f"  Min width: {self.width_statistics.get('min', 0):.4f} nm\n")
                f.write(f"  Max width: {self.width_statistics.get('max', 0):.4f} nm\n\n")
                
            # Fin pitch statistics
            if hasattr(self, 'pitch_statistics') and self.pitch_statistics.get('count', 0) > 0:
                f.write(f"Fin Pitch Statistics:\n")
                f.write(f"  Number of measurements: {self.pitch_statistics.get('count', 0)}\n")
                f.write(f"  Mean pitch: {self.pitch_statistics.get('mean', 0):.4f} nm\n")
                f.write(f"  Standard deviation: {self.pitch_statistics.get('std', 0):.4f} nm\n")
                f.write(f"  Min pitch: {self.pitch_statistics.get('min', 0):.4f} nm\n")
                f.write(f"  Max pitch: {self.pitch_statistics.get('max', 0):.4f} nm\n\n")
                
            # Fin height statistics
            if hasattr(self, 'height_statistics') and self.height_statistics.get('count', 0) > 0:
                f.write(f"Fin Height Statistics:\n")
                f.write(f"  Number of measurements: {self.height_statistics.get('count', 0)}\n")
                f.write(f"  Mean height: {self.height_statistics.get('mean', 0):.4f} nm\n")
                f.write(f"  Standard deviation: {self.height_statistics.get('std', 0):.4f} nm\n")
                f.write(f"  Min height: {self.height_statistics.get('min', 0):.4f} nm\n")
                f.write(f"  Max height: {self.height_statistics.get('max', 0):.4f} nm\n\n")
                
            # Coating statistics if available
            if hasattr(self, 'coating_statistics') and self.coating_statistics is not None and self.coating_statistics.get('count', 0) > 0:
                f.write(f"Coating Thickness Statistics:\n")
                f.write(f"  Number of coatings: {self.coating_statistics.get('count', 0)}\n")
                f.write(f"  Mean thickness: {self.coating_statistics.get('mean', 0):.4f} nm\n")
                f.write(f"  Standard deviation: {self.coating_statistics.get('std', 0):.4f} nm\n")
                f.write(f"  Min thickness: {self.coating_statistics.get('min', 0):.4f} nm\n")
                f.write(f"  Max thickness: {self.coating_statistics.get('max', 0):.4f} nm\n")
                
        # Save text summary
        summary_file = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(self.results.get_summary())
            
        # Save visualization
        # Use the visualize method to get the figure and axes
        fig, _ = self.visualize(show=False)
            
        plot_file = os.path.join(output_dir, f"{base_name}_visualization.{plot_format}")
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f"Results saved to directory: {output_dir}")
        return output_dir 