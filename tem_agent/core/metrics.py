#!/usr/bin/env python3
"""Core metrics module for TEM image analysis measurements."""

import numpy as np
from scipy import ndimage
from skimage import measure

class Metrics:
    """Static class for TEM image measurement functions."""
    
    @staticmethod
    def measure_distance(point1, point2, scale_factor=1.0):
        """Measure the Euclidean distance between two points.
        
        Args:
            point1: First point (y, x)
            point2: Second point (y, x)
            scale_factor: Scale factor for physical units (nm/pixel)
            
        Returns:
            Distance in physical units or pixels
        """
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) * scale_factor
    
    @staticmethod
    def measure_profile(image, start_point, end_point, width=1, mode='perpendicular'):
        """Extract an intensity profile between two points.
        
        Args:
            image: Input image
            start_point: Starting point (y, x)
            end_point: Ending point (y, x)
            width: Width of the profile line in pixels
            mode: 'perpendicular' or 'along' for orientation
            
        Returns:
            Tuple of (profile intensity array, distances array)
        """
        from skimage.measure import profile_line
        
        # Extract the profile
        profile = profile_line(image, start_point, end_point, linewidth=width, mode='reflect')
        
        # Calculate the distances along the profile
        total_distance = Metrics.measure_distance(start_point, end_point)
        distances = np.linspace(0, total_distance, len(profile))
        
        return profile, distances
    
    @staticmethod
    def measure_region_properties(labeled_image, intensity_image=None, properties=None):
        """Measure properties of labeled regions in an image.
        
        Args:
            labeled_image: Labeled image (integer labels for each region)
            intensity_image: Original intensity image (optional)
            properties: List of properties to measure
            
        Returns:
            List of region property dictionaries
        """
        if properties is None:
            properties = ['area', 'centroid', 'bbox', 'orientation', 'major_axis_length', 'minor_axis_length']
            
        # Measure region properties
        regions = measure.regionprops(labeled_image, intensity_image=intensity_image)
        
        # Extract requested properties for each region
        region_props = []
        for region in regions:
            props = {}
            for prop in properties:
                if hasattr(region, prop):
                    props[prop] = getattr(region, prop)
            region_props.append(props)
            
        return region_props
    
    @staticmethod
    def find_interfaces(labeled_image):
        """Find interfaces between different regions in a labeled image.
        
        Args:
            labeled_image: Labeled image (integer labels for each region)
            
        Returns:
            Binary image with interfaces marked
        """
        # Initialize an empty image for interfaces
        interfaces = np.zeros_like(labeled_image, dtype=bool)
        
        # Iterate through each region
        for region_id in np.unique(labeled_image):
            if region_id == 0:  # Skip background
                continue
                
            # Create a binary mask for this region
            region_mask = labeled_image == region_id
            
            # Dilate the mask
            dilated = ndimage.binary_dilation(region_mask)
            
            # The interface is the difference between the dilated mask and the original
            region_interface = dilated & ~region_mask
            
            # Add this region's interface to the overall interface image
            interfaces |= region_interface
            
        return interfaces
    
    @staticmethod
    def measure_thickness(labeled_image, scale_factor=1.0, axis=0):
        """Measure thickness of a layer along a specified axis.
        
        Args:
            labeled_image: Labeled image (integer labels for each region)
            scale_factor: Scale factor for physical units (nm/pixel)
            axis: Axis along which to measure thickness (0=vertical, 1=horizontal)
            
        Returns:
            Dictionary with thickness measurements
        """
        # Identify unique regions
        regions = np.unique(labeled_image)
        regions = regions[regions > 0]  # Remove background
        
        if len(regions) < 2:
            raise ValueError("Need at least two regions to measure thickness")
            
        # Initialize results
        measurements = []
        
        # For each non-background region
        for region in regions:
            # Create binary mask for this region
            mask = labeled_image == region
            
            # Get thickness profile along specified axis
            if axis == 0:  # Vertical
                profile = np.sum(mask, axis=1)  # Sum along rows
            else:  # Horizontal
                profile = np.sum(mask, axis=0)  # Sum along columns
                
            # Find where the profile is non-zero
            non_zero = np.where(profile > 0)[0]
            
            if len(non_zero) > 0:
                # Measure thickness
                thickness = (non_zero.max() - non_zero.min() + 1) * scale_factor
                
                # Record measurement
                measurements.append({
                    'region': int(region),
                    'thickness': thickness,
                    'start': int(non_zero.min()),
                    'end': int(non_zero.max())
                })
                
        return measurements
    
    @staticmethod
    def calculate_statistics(measurements, key='thickness'):
        """Calculate statistics for a set of measurements.
        
        Args:
            measurements: List of measurement dictionaries
            key: Key in the dictionaries to analyze
            
        Returns:
            Dictionary with statistics
        """
        values = [m[key] for m in measurements if key in m]
        
        if not values:
            return {
                'count': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None
            }
            
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        } 