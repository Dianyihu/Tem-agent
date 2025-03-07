import os
import numpy as np
import matplotlib.pyplot as plt

from tem_agent.utils.io import load_dm3, save_results, save_plot
from tem_agent.utils.image_processing import (preprocess_image, detect_edges, 
                                              auto_threshold, find_interfaces,
                                              calibrate_scale, plot_results)
from tem_agent.utils.thickness import (measure_film_thickness, measure_film_profile,
                                       detect_layers)

class TemAnalyzer:
    """Main class for analyzing TEM images and measuring thin film thickness."""
    
    def __init__(self, pixel_size=None):
        """Initialize the analyzer.
        
        Args:
            pixel_size: Physical size per pixel in nm/pixel, if known
        """
        self.image = None
        self.metadata = None
        self.labeled_image = None
        self.measurements = None
        self.mean_thickness = None
        self.std_thickness = None
        self.pixel_size = pixel_size
        self.scale_factor = 1.0  # Will be updated if calibration info is available
        self.filename = None
    
    def load_image(self, file_path):
        """Load a TEM image from a .dm3 file."""
        self.filename = file_path
        data, metadata, scale_factors = load_dm3(file_path)
        
        if data is None:
            raise ValueError(f"Failed to load image from {file_path}")
            
        self.image = data
        self.metadata = metadata
        
        # Set scale factor if available from metadata
        if scale_factors is not None:
            self.scale_factor = scale_factors[0]  # Use x-scale as default
            print(f"Loaded image with scale factor: {self.scale_factor} nm/pixel")
        elif self.pixel_size is not None:
            self.scale_factor = self.pixel_size
            print(f"Using provided pixel size: {self.scale_factor} nm/pixel")
        
        print(f"Loaded image with shape: {self.image.shape}")
        return self.image
    
    def preprocess(self, sigma=1.0):
        """Preprocess the loaded image."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        self.processed_image = preprocess_image(self.image, sigma=sigma)
        return self.processed_image
    
    def detect_film_substrate_interface(self, edge_method='sobel', threshold_method='otsu', min_region_size=100):
        """Detect the interface between thin film and substrate."""
        if not hasattr(self, 'processed_image'):
            self.preprocess()
        
        # Detect edges
        self.edges = detect_edges(self.processed_image, method=edge_method)
        
        # Threshold the image to separate regions
        self.binary_image, self.threshold = auto_threshold(self.processed_image, method=threshold_method)
        
        # Find interfaces between regions
        self.labeled_image, self.regions, self.boundaries = find_interfaces(self.binary_image, min_size=min_region_size)
        
        return self.labeled_image
    
    def measure_thickness(self):
        """Measure the thickness of the thin film."""
        if self.labeled_image is None:
            self.detect_film_substrate_interface()
        
        # Measure film thickness
        self.measurements, self.mean_thickness, self.std_thickness = measure_film_thickness(
            self.labeled_image, scale_factor=self.scale_factor
        )
        
        if not self.measurements:
            print("Warning: No film thickness measurements could be obtained. "
                  "Try adjusting preprocessing parameters.")
        else:
            print(f"Mean film thickness: {self.mean_thickness:.2f} nm ± {self.std_thickness:.2f} nm")
            
        return self.measurements, self.mean_thickness, self.std_thickness
    
    def visualize_results(self):
        """Visualize the analysis results."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Make sure we have all the necessary data
        if not hasattr(self, 'edges'):
            self.detect_film_substrate_interface()
        
        if not hasattr(self, 'measurements'):
            self.measure_thickness()
        
        # Plot results
        self.fig = plot_results(
            self.image, 
            edges=self.edges, 
            boundaries=self.boundaries,
            measurements=self.measurements
        )
        
        return self.fig
    
    def save_results(self, output_dir="results"):
        """Save analysis results to files."""
        if self.image is None or self.filename is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Make sure we have measurements
        if not hasattr(self, 'measurements'):
            self.measure_thickness()
        
        # Save numerical results
        save_results(
            output_dir, 
            self.filename, 
            self.labeled_image, 
            self.measurements, 
            self.mean_thickness,
            self.std_thickness
        )
        
        # Save visualization if available
        if hasattr(self, 'fig'):
            save_plot(self.fig, output_dir, self.filename)
        else:
            fig = self.visualize_results()
            save_plot(fig, output_dir, self.filename)
        
        print(f"Results saved to {output_dir}")
    
    def get_thickness_stats(self):
        """Get thickness statistics."""
        if not hasattr(self, 'mean_thickness'):
            self.measure_thickness()
            
        return self.mean_thickness, self.std_thickness
    
    def advanced_layer_detection(self, min_size=100, max_regions=5):
        """Perform advanced layer detection for complex multilayer films."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        if not hasattr(self, 'processed_image'):
            self.preprocess()
            
        # Detect layers using more advanced methods
        self.labeled_layers, self.layer_regions = detect_layers(
            self.processed_image, 
            min_size=min_size,
            max_regions=max_regions
        )
        
        # Update labeled image for subsequent thickness measurements
        self.labeled_image = self.labeled_layers
        
        return self.labeled_layers 