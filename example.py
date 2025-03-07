#!/usr/bin/env python3
"""
Example script demonstrating how to use the TEM Image Analyzer in a Python script.
"""

import os
import matplotlib.pyplot as plt
from tem_agent import TemAnalyzer

def main():
    """Demonstrate the usage of the TEM Analyzer."""
    # Path to your DM3 file (replace with your actual file path)
    dm3_file = "path/to/your/sample.dm3"
    
    # Check if the file exists
    if not os.path.exists(dm3_file):
        print(f"File {dm3_file} does not exist.")
        print("Please replace the file path in this example with your actual .dm3 file.")
        return
    
    # Initialize the analyzer with optional pixel size (in nm/pixel)
    # If not provided, it will try to extract from file metadata
    analyzer = TemAnalyzer(pixel_size=None)
    
    # Load the image
    print(f"Loading image: {dm3_file}")
    analyzer.load_image(dm3_file)
    
    # Preprocess the image with Gaussian smoothing
    print("Preprocessing image...")
    analyzer.preprocess(sigma=1.0)
    
    # Method 1: Standard interface detection
    print("Detecting film-substrate interface...")
    analyzer.detect_film_substrate_interface(
        edge_method='sobel',
        threshold_method='otsu',
        min_region_size=100
    )
    
    # Method 2 (alternative): Advanced layer detection
    # Uncomment to use this instead of the standard method
    # print("Performing advanced layer detection...")
    # analyzer.advanced_layer_detection(min_size=100, max_regions=5)
    
    # Measure film thickness
    print("Measuring film thickness...")
    measurements, mean_thickness, std_thickness = analyzer.measure_thickness()
    
    # Print results
    print("\nResults:")
    print(f"Mean film thickness: {mean_thickness:.2f} nm")
    print(f"Standard deviation: {std_thickness:.2f} nm")
    print(f"Relative variation: {(std_thickness/mean_thickness*100):.2f}%")
    print(f"Number of measurement points: {len(measurements)}")
    
    # Visualize results
    print("\nGenerating visualization...")
    fig = analyzer.visualize_results()
    
    # Save results
    output_dir = "example_results"
    print(f"\nSaving results to {output_dir}...")
    analyzer.save_results(output_dir=output_dir)
    
    # Show the visualization
    plt.show()

if __name__ == "__main__":
    main() 