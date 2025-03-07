#!/usr/bin/env python3
"""
Example script demonstrating how to use the FINFET Analyzer in a Python script.
"""

import os
import matplotlib.pyplot as plt
from tem_agent import FinfetAnalyzer

def main():
    """Demonstrate the usage of the FINFET Analyzer."""
    # Path to your JPEG file (replace with your actual file path)
    image_file = "finfet.jpeg"
    
    # Check if the file exists
    if not os.path.exists(image_file):
        print(f"File {image_file} does not exist.")
        print("Please replace the file path in this example with your actual FINFET image file.")
        return
    
    # Initialize the analyzer with optional pixel size (in nm/pixel)
    # If known, provide the pixel size for accurate physical measurements
    analyzer = FinfetAnalyzer(pixel_size=None)  # Set to actual value if known, e.g., 0.5
    
    # Load the image
    print(f"Loading image: {image_file}")
    analyzer.load_image(image_file)
    
    # Step 1: Detect fin structures
    print("\nDetecting fin structures...")
    fin_regions = analyzer.detect_fins(
        sigma=2.0,              # Gaussian smoothing parameter
        thresh_method='otsu',   # Thresholding method (otsu, yen, mean)
        min_size=100,           # Minimum region size in pixels
        max_regions=10          # Maximum number of regions to detect
    )
    
    # Step 2: Measure fin widths
    print("\nMeasuring fin widths...")
    fin_widths = analyzer.measure_fins()
    
    # Step 3: Detect coating
    print("\nDetecting coating on fins...")
    coating_edges, coating_aoi = analyzer.detect_coating(
        sigma=1.0,          # Gaussian smoothing parameter
        edge_method='canny' # Edge detection method (canny, sobel, scharr)
    )
    
    # Step 4: Measure coating thickness
    print("\nMeasuring coating thickness...")
    coating_measurements = analyzer.measure_coating(scaled=True)
    
    # Step 5: Visualize results
    print("\nGenerating visualization...")
    fig = analyzer.visualize_results()
    
    # Step 6: Save results
    output_dir = "finfet_results"
    print(f"\nSaving results to {output_dir}...")
    analyzer.save_results(output_dir=output_dir)
    
    # Alternative: Run the complete analysis pipeline in one step
    # This is equivalent to steps 1-6 above
    # analyzer.run_full_analysis(
    #     sigma_fins=2.0,
    #     thresh_method='otsu',
    #     min_size=100,
    #     sigma_coating=1.0,
    #     edge_method='canny',
    #     save_dir="finfet_results"
    # )
    
    # Show the visualization
    plt.show()

if __name__ == "__main__":
    main() 