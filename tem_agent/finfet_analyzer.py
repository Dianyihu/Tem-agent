import os
import numpy as np
import matplotlib.pyplot as plt

from tem_agent.utils.io import load_image, save_finfet_results, save_plot
from tem_agent.utils.finfet import (detect_fin_structures, measure_fin_width,
                                   detect_fin_coating, measure_coating_thickness,
                                   visualize_finfet_results)

class FinfetAnalyzer:
    """Class for analyzing FINFET structures in images."""
    
    def __init__(self, pixel_size=None, u_shaped=False):
        """Initialize the FINFET analyzer.
        
        Args:
            pixel_size: Physical size per pixel in nm/pixel, if known
            u_shaped: Set to True if analyzing U-shaped FINFET structures
        """
        self.image = None
        self.fin_mask = None
        self.fin_regions = None
        self.fin_widths = None
        self.coating_edges = None
        self.coating_aoi = None
        self.coating_measurements = None
        self.filename = None
        self.pixel_size = pixel_size
        self.scale_factor = 1.0  # Default scale factor (1 pixel = 1 unit)
        self.u_shaped = u_shaped  # Flag for U-shaped fin detection
    
    def load_image(self, file_path):
        """Load an image file (JPEG, PNG, etc.) containing FINFET structures."""
        self.filename = file_path
        image, metadata, scale_factors = load_image(file_path)
        
        if image is None:
            raise ValueError(f"Failed to load image from {file_path}")
        
        self.image = image
        
        # Set scale factor if available
        if scale_factors is not None:
            self.scale_factor = scale_factors[0]
            print(f"Loaded image with scale factor: {self.scale_factor} nm/pixel")
        elif self.pixel_size is not None:
            self.scale_factor = self.pixel_size
            print(f"Using provided pixel size: {self.scale_factor} nm/pixel")
        
        # Try to extract scale from image if it contains a scale bar
        # This is just a placeholder - would need more advanced code to detect scale bar
        if self.scale_factor == 1.0:
            try:
                # Check if the image has a scale bar text (e.g., "10 nm")
                # This is a very simple approach that just checks for common scale indicators
                scale_extracted = False
                if hasattr(self.image, 'dtype') and self.image.dtype == np.uint8:
                    # Look for a scale bar in the bottom left corner (common location)
                    # This is just a placeholder - real implementation would need OCR or similar
                    pass
                
                # If found scale text, you could set self.scale_factor accordingly
                if scale_extracted:
                    print(f"Extracted scale factor from image: {self.scale_factor} nm/pixel")
            except:
                pass  # Silently continue if scale extraction fails
                
        print(f"Loaded image with shape: {self.image.shape}")
        return self.image
    
    def detect_fins(self, sigma=2.0, thresh_method='otsu', min_size=100, max_regions=10):
        """Detect FIN structures in the image."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        self.fin_mask, self.fin_regions = detect_fin_structures(
            self.image, 
            sigma=sigma,
            thresh_method=thresh_method,
            min_size=min_size,
            max_regions=max_regions,
            u_shaped=self.u_shaped  # Pass the U-shaped flag
        )
        
        print(f"Detected {len(self.fin_regions)} fin structures")
        return self.fin_regions
    
    def measure_fins(self, num_profiles=10):
        """Measure the width of detected fins."""
        if self.fin_regions is None:
            self.detect_fins()
        
        # Pass the original image to the updated measure_fin_width function
        self.fin_widths = measure_fin_width(
            self.fin_regions, 
            self.image,
            u_shaped=self.u_shaped,
            num_profiles=num_profiles,
            scale_factor=self.scale_factor
        )
        
        # Print results
        print("\nFin width measurements:")
        for i, (region, width, _) in enumerate(self.fin_widths):
            scaled_width = width * self.scale_factor
            unit = "nm" if self.scale_factor != 1.0 else "px"
            print(f"Fin {i+1}: {scaled_width:.2f} {unit}")
        
        return self.fin_widths
    
    def detect_coating(self, sigma=1.0, edge_method='canny'):
        """Detect coating on fin structures."""
        if self.fin_mask is None:
            self.detect_fins()
        
        self.coating_edges, self.coating_aoi = detect_fin_coating(
            self.image,
            self.fin_mask,
            sigma=sigma,
            edge_method=edge_method
        )
        
        return self.coating_edges, self.coating_aoi
    
    def measure_coating(self, scaled=True):
        """Measure the thickness of the coating around fins."""
        if self.coating_aoi is None:
            self.detect_coating()
        
        self.coating_measurements = measure_coating_thickness(
            self.fin_regions,
            self.coating_aoi,
            scaled=scaled,
            scale_factor=self.scale_factor
        )
        
        # Print results
        print("\nCoating thickness measurements:")
        for i, coating in enumerate(self.coating_measurements):
            avg = coating['avg_thickness']
            std = coating['std_thickness']
            unit = "nm" if scaled and self.scale_factor != 1.0 else "px"
            print(f"Fin {i+1} coating: {avg:.2f} ± {std:.2f} {unit}")
        
        return self.coating_measurements
    
    def visualize_results(self):
        """Visualize the analysis results."""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Make sure we have measurements
        if self.fin_widths is None:
            self.measure_fins()
        
        # Optionally detect coating if not already done
        if not hasattr(self, 'coating_measurements') or self.coating_measurements is None:
            try:
                self.measure_coating()
            except:
                print("Could not measure coating. Visualizing fin measurements only.")
        
        # Generate visualization
        self.fig = visualize_finfet_results(
            self.image,
            self.fin_regions,
            self.fin_widths,
            self.coating_measurements
        )
        
        return self.fig
    
    def save_results(self, output_dir="results"):
        """Save the analysis results to files."""
        if self.image is None or self.filename is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Make sure we have measurements
        if self.fin_widths is None:
            self.measure_fins()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save fin width measurements to CSV
        base_name = os.path.splitext(os.path.basename(self.filename))[0]
        with open(os.path.join(output_dir, f"{base_name}_fin_widths.csv"), 'w') as f:
            f.write("fin_id,width_px,width_nm\n")
            for i, (region, width, _) in enumerate(self.fin_widths):
                scaled_width = width * self.scale_factor
                f.write(f"{i+1},{width:.4f},{scaled_width:.4f}\n")
        
        # Save summary text file
        with open(os.path.join(output_dir, f"{base_name}_summary.txt"), 'w') as f:
            f.write(f"Number of fins detected: {len(self.fin_widths)}\n")
            for i, (region, width, _) in enumerate(self.fin_widths):
                scaled_width = width * self.scale_factor
                f.write(f"Fin {i+1} width: {width:.4f} px = {scaled_width:.4f} nm\n")
        
        # Save visualization if available
        if hasattr(self, 'fig'):
            save_plot(self.fig, output_dir, self.filename)
        else:
            fig = self.visualize_results()
            save_plot(fig, output_dir, self.filename)
            
        print(f"Results saved to {output_dir}")
    
    def run_full_analysis(self, sigma_fins=2.0, thresh_method='otsu', min_size=100, 
                         sigma_coating=1.0, edge_method='canny', save_dir="results"):
        """Run the complete analysis pipeline."""
        # Detect fins
        self.detect_fins(sigma=sigma_fins, thresh_method=thresh_method, min_size=min_size)
        
        # Measure fin widths
        self.measure_fins()
        
        # Detect and measure coating
        self.detect_coating(sigma=sigma_coating, edge_method=edge_method)
        self.measure_coating()
        
        # Visualize results
        self.visualize_results()
        
        # Save results
        self.save_results(output_dir=save_dir)
        
        return self.fin_widths, self.coating_measurements
        
    def try_different_parameters(self, min_sizes=[100, 200, 500, 1000], 
                                sigmas=[1.0, 2.0, 3.0], 
                                thresholds=['otsu', 'yen', 'mean']):
        """Try different parameter combinations to find the best result."""
        best_params = None
        max_fins = 0
        
        for min_size in min_sizes:
            for sigma in sigmas:
                for thresh in thresholds:
                    print(f"Trying: min_size={min_size}, sigma={sigma}, threshold={thresh}")
                    try:
                        regions = self.detect_fins(sigma=sigma, 
                                                 thresh_method=thresh, 
                                                 min_size=min_size)
                        num_fins = len(regions)
                        
                        if num_fins > max_fins:
                            max_fins = num_fins
                            best_params = {
                                'min_size': min_size,
                                'sigma': sigma,
                                'threshold': thresh,
                                'num_fins': num_fins
                            }
                            print(f"New best: {num_fins} fins detected!")
                    except Exception as e:
                        print(f"Error with these parameters: {str(e)}")
        
        # Apply the best parameters
        if best_params:
            print("\nBest parameters found:")
            print(f"min_size={best_params['min_size']}, "
                  f"sigma={best_params['sigma']}, "
                  f"threshold={best_params['threshold']}")
            
            self.detect_fins(sigma=best_params['sigma'],
                           thresh_method=best_params['threshold'],
                           min_size=best_params['min_size'])
            self.measure_fins()
            
            return best_params
        
        return None 