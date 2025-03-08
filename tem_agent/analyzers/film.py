#!/usr/bin/env python3
"""Film thickness analyzer for TEM images."""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, measure, morphology, filters

from tem_agent.core import BaseAnalyzer, AnalysisResult, ImageHandler, Metrics

class FilmResult(AnalysisResult):
    """Class for film thickness analysis results."""
    
    def _get_text_summary(self):
        """Generate a text summary of film thickness results."""
        if not self.data or not self.metadata:
            return "No film thickness measurements available."
            
        summary = f"Film Thickness Analysis Results\n"
        summary += f"==============================\n\n"
        
        # Add image info
        if 'image_info' in self.metadata:
            info = self.metadata['image_info']
            summary += f"Image: {info.get('filename', 'Unknown')}\n"
            summary += f"Dimensions: {info.get('shape', 'Unknown')}\n"
            summary += f"Scale: {info.get('scale_factor', 'Unknown')}\n\n"
        
        # Add statistics
        if 'statistics' in self.metadata:
            stats = self.metadata['statistics']
            summary += f"Mean thickness: {stats.get('mean', 0):.2f} nm\n"
            summary += f"Standard deviation: {stats.get('std', 0):.2f} nm\n"
            summary += f"Min thickness: {stats.get('min', 0):.2f} nm\n"
            summary += f"Max thickness: {stats.get('max', 0):.2f} nm\n"
            summary += f"Measurements: {stats.get('count', 0)}\n\n"
            
        # Add individual measurements
        summary += "Detailed Measurements:\n"
        for i, m in enumerate(self.data):
            summary += f"  Region {m.get('region', i+1)}: {m.get('thickness', 0):.2f} nm\n"
            
        return summary


class FilmAnalyzer(BaseAnalyzer):
    """Analyzer for measuring thin film thickness in TEM images."""
    
    def __init__(self, pixel_size=None):
        """Initialize film analyzer.
        
        Args:
            pixel_size: Physical size per pixel in nm/pixel, if known
        """
        super().__init__(pixel_size=pixel_size)
        self.labeled_image = None
        self.edges = None
        self.regions = None
        self.boundaries = None
        self.binary_image = None
        self.threshold = None
        self.measurements = None
        self.statistics = None
        
    def load_image(self, file_path):
        """Load a TEM image from a file."""
        self.image, self.metadata, scale_factors = ImageHandler.load_image(file_path)
        self.filename = file_path
        
        # Additional processing for SEM images
        if self.image is not None:
            # Check if this is likely an SEM image with info bar
            height, width = self.image.shape[:2]
            
            # If the image has a high aspect ratio, it might be a dual-panel image
            if width > height * 1.5:
                # This might be a side-by-side comparison (original + analysis)
                midpoint = width // 2
                self.original_image = self.image[:, :midpoint]
                self.analysis_image = self.image[:, midpoint:]
                
                # Use only the original image for processing
                self.image = self.original_image
                print("Detected dual-panel image, using left panel for analysis")
        
        # Set scale factor if available from metadata
        if scale_factors is not None:
            self.scale_factor = scale_factors[0]  # Use x-scale as default
            print(f"Loaded image with scale factor: {self.scale_factor} nm/pixel")
        elif self.pixel_size is not None:
            self.scale_factor = self.pixel_size
            print(f"Using provided pixel size: {self.scale_factor} nm/pixel")
            
        return self.image
    
    def preprocess(self, sigma=1.0, denoise=False, normalize=True, **kwargs):
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
    
    def detect_film_substrate_interface(self, edge_method='sobel', threshold_method='otsu', min_region_size=100):
        """Detect the interface between thin film and substrate."""
        if not hasattr(self, 'processed_image') or self.processed_image is None:
            self.preprocess()
        
        # Detect edges
        self.edges = ImageHandler.detect_edges(self.processed_image, method=edge_method)
        
        # Threshold the image to separate regions
        from skimage.filters import threshold_otsu, threshold_yen, threshold_mean
        
        if threshold_method == 'otsu':
            self.threshold = threshold_otsu(self.processed_image)
        elif threshold_method == 'yen':
            self.threshold = threshold_yen(self.processed_image)
        elif threshold_method == 'mean':
            self.threshold = threshold_mean(self.processed_image)
        else:
            raise ValueError(f"Unsupported threshold method: {threshold_method}")
            
        self.binary_image = self.processed_image > self.threshold
        
        # Remove small regions
        self.binary_image = morphology.remove_small_objects(self.binary_image, min_size=min_region_size)
        
        # Additional step to remove potential SEM info bar regions
        # Info bars typically have text which creates many small connected components
        height, width = self.binary_image.shape
        bottom_region = self.binary_image[int(height*0.9):, :]
        
        # If the bottom region has many small components, it might be an info bar
        labeled_bottom, num_features = measure.label(bottom_region, return_num=True)
        if num_features > width / 20:  # Heuristic: many small features indicate text
            print(f"Detected potential SEM info bar with {num_features} features, excluding from analysis")
            self.binary_image[int(height*0.9):, :] = False
        
        # Label regions
        self.labeled_image, num_regions = measure.label(self.binary_image, return_num=True)
        
        # Get region properties
        self.regions = Metrics.measure_region_properties(self.labeled_image)
        
        # Find interfaces
        self.boundaries = Metrics.find_interfaces(self.labeled_image)
        
        return self.labeled_image
    
    def advanced_layer_detection(self, min_size=100, max_regions=5):
        """Perform advanced layer detection for complex multilayer films."""
        if not hasattr(self, 'processed_image') or self.processed_image is None:
            self.preprocess()
            
        # Use watershed segmentation for more advanced layer detection
        from skimage.morphology import disk
        from skimage.filters import rank
        from skimage.segmentation import watershed
        
        # Calculate gradients
        gradient = rank.gradient(self.processed_image, disk(2))
        
        # Mark seeds for watershed
        markers = np.zeros_like(self.processed_image, dtype=np.uint)
        
        # Automatic seed detection using local minima
        from skimage.feature import peak_local_max
        
        # Find local minima
        local_min = peak_local_max(-self.processed_image, 
                                  min_distance=20, 
                                  exclude_border=False, 
                                  indices=False)
        
        # Label seeds
        markers, _ = measure.label(local_min, return_num=True)
        
        # Apply watershed segmentation
        self.labeled_image = watershed(gradient, markers)
        
        # Remove small regions
        for region_id in np.unique(self.labeled_image):
            if region_id == 0:  # Skip background
                continue
                
            # Count pixels in this region
            if np.sum(self.labeled_image == region_id) < min_size:
                self.labeled_image[self.labeled_image == region_id] = 0
                
        # Re-label to ensure consecutive labels
        self.labeled_image, _ = measure.label(self.labeled_image > 0, return_num=True)
        
        # Get region properties
        self.regions = Metrics.measure_region_properties(self.labeled_image)
        
        # Find interfaces
        self.boundaries = Metrics.find_interfaces(self.labeled_image)
        
        return self.labeled_image
    
    def analyze(self, method='basic', **kwargs):
        """Analyze film thickness.
        
        Args:
            method: Analysis method ('basic' or 'advanced')
            **kwargs: Additional parameters for the method
            
        Returns:
            FilmResult object with thickness measurements
        """
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        # Select analysis method
        if method == 'basic':
            edge_method = kwargs.get('edge_method', 'sobel')
            threshold_method = kwargs.get('threshold_method', 'otsu')
            min_region_size = kwargs.get('min_region_size', 100)
            
            self.detect_film_substrate_interface(
                edge_method=edge_method,
                threshold_method=threshold_method,
                min_region_size=min_region_size
            )
        elif method == 'advanced':
            min_size = kwargs.get('min_size', 100)
            max_regions = kwargs.get('max_regions', 5)
            
            self.advanced_layer_detection(
                min_size=min_size,
                max_regions=max_regions
            )
        else:
            raise ValueError(f"Unsupported analysis method: {method}")
        
        # Measure thickness
        axis = kwargs.get('axis', 0)  # Default to vertical
        self.measurements = Metrics.measure_thickness(
            self.labeled_image, 
            scale_factor=self.scale_factor,
            axis=axis
        )
        
        # Calculate statistics
        self.statistics = Metrics.calculate_statistics(self.measurements, key='thickness')
        
        # Create result object
        result = FilmResult(
            data=self.measurements,
            metadata={
                'statistics': self.statistics,
                'image_info': self.get_image_info(),
                'parameters': kwargs
            }
        )
        
        self.results = result
        return result
    
    def visualize(self, show=True, figsize=(12, 8)):
        """Visualize film thickness analysis results."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        if not hasattr(self, 'measurements') or not self.measurements:
            self.analyze()
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Processed image
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            axes[1].imshow(self.processed_image, cmap='gray')
            axes[1].set_title('Processed Image')
        else:
            axes[1].imshow(self.image, cmap='gray')
            axes[1].set_title('Original Image')
        axes[1].axis('off')
        
        # Labeled regions
        if hasattr(self, 'labeled_image') and self.labeled_image is not None:
            from skimage.color import label2rgb
            color_labels = label2rgb(self.labeled_image, image=self.image, bg_label=0)
            axes[2].imshow(color_labels)
            axes[2].set_title('Labeled Regions')
        else:
            axes[2].imshow(self.image, cmap='gray')
            axes[2].set_title('No Regions Detected')
        axes[2].axis('off')
        
        # Measurements visualization
        axes[3].imshow(self.image, cmap='gray')
        axes[3].set_title('Film Thickness Measurements')
        
        if hasattr(self, 'measurements') and self.measurements:
            for m in self.measurements:
                # Draw measurement line
                if 'start' in m and 'end' in m:
                    thickness = m.get('thickness', 0)
                    region = m.get('region', 0)
                    
                    # For vertical measurements
                    mid_x = self.image.shape[1] // 2
                    start_y = m.get('start', 0)
                    end_y = m.get('end', start_y)
                    
                    # Draw line
                    axes[3].plot([mid_x, mid_x], [start_y, end_y], 'r-', linewidth=2)
                    
                    # Add text
                    axes[3].text(mid_x + 10, (start_y + end_y) // 2, 
                                f'R{region}: {thickness:.1f} nm', 
                                color='white', fontsize=10,
                                bbox=dict(facecolor='red', alpha=0.7))
        
        axes[3].axis('off')
        
        # Add overall title with statistics
        if hasattr(self, 'statistics') and self.statistics:
            mean = self.statistics.get('mean', 0)
            std = self.statistics.get('std', 0)
            plt.suptitle(f'Film Thickness Analysis: {mean:.2f} ± {std:.2f} nm', fontsize=16)
        else:
            plt.suptitle('Film Thickness Analysis', fontsize=16)
            
        plt.tight_layout()
        
        if show:
            plt.show()
            
        self.fig = fig
        return fig
    
    def save_results(self, output_dir="results", plot_format='png'):
        """Save analysis results to files."""
        if self.results is None:
            raise ValueError("No results to save. Call analyze() first.")
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        
        # Save measurements as CSV
        measurements_file = os.path.join(output_dir, f"{base_name}_measurements.csv")
        with open(measurements_file, 'w') as f:
            f.write("region,thickness_nm,start_pos,end_pos\n")
            for m in self.measurements:
                f.write(f"{m.get('region', 0)},{m.get('thickness', 0):.4f},{m.get('start', 0)},{m.get('end', 0)}\n")
                
        # Save statistics
        stats_file = os.path.join(output_dir, f"{base_name}_statistics.txt")
        if self.statistics:
            with open(stats_file, 'w') as f:
                f.write(f"Film Thickness Statistics\n")
                f.write(f"=====================\n\n")
                f.write(f"Mean thickness: {self.statistics.get('mean', 0):.4f} nm\n")
                f.write(f"Standard deviation: {self.statistics.get('std', 0):.4f} nm\n")
                f.write(f"Min thickness: {self.statistics.get('min', 0):.4f} nm\n")
                f.write(f"Max thickness: {self.statistics.get('max', 0):.4f} nm\n")
                f.write(f"Number of measurements: {self.statistics.get('count', 0)}\n")
                
        # Save text summary
        summary_file = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(self.results.get_summary())
            
        # Save visualization
        if not hasattr(self, 'fig'):
            self.visualize(show=False)
            
        plot_file = os.path.join(output_dir, f"{base_name}_visualization.{plot_format}")
        self.fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f"Results saved to directory: {output_dir}")
        return output_dir