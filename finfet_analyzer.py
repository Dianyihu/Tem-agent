#!/usr/bin/env python3
"""
FINFET Image Analyzer - Command line tool for analyzing FINFET structures in images.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from tem_agent import FinfetAnalyzer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze FINFET structures in images and measure fin width and coating thickness."
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input image file (JPEG, PNG) or directory containing images"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="results",
        help="Directory to save analysis results (default: results)"
    )
    
    parser.add_argument(
        "--pixel-size", "-p", 
        type=float,
        help="Pixel size in nm/pixel for physical measurements"
    )
    
    parser.add_argument(
        "--fin-sigma", "-fs", 
        type=float, 
        default=2.0,
        help="Sigma value for Gaussian smoothing in fin detection (default: 2.0)"
    )
    
    parser.add_argument(
        "--threshold-method", "-t", 
        choices=["otsu", "yen", "mean"], 
        default="otsu",
        help="Thresholding method for fin detection (default: otsu)"
    )
    
    parser.add_argument(
        "--min-size", "-m", 
        type=int, 
        default=100,
        help="Minimum region size in pixels for fin detection (default: 100)"
    )
    
    parser.add_argument(
        "--max-regions", "-mr", 
        type=int, 
        default=10,
        help="Maximum number of fin regions to detect (default: 10)"
    )
    
    parser.add_argument(
        "--coating-sigma", "-cs", 
        type=float, 
        default=1.0,
        help="Sigma value for coating edge detection (default: 1.0)"
    )
    
    parser.add_argument(
        "--edge-method", "-e", 
        choices=["canny", "sobel", "scharr"], 
        default="canny",
        help="Edge detection method for coating (default: canny)"
    )
    
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show visualization plots during analysis"
    )
    
    return parser.parse_args()

def process_single_file(file_path, args):
    """Process a single image file."""
    print(f"Processing file: {file_path}")
    
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File {file_path} does not exist")
        return None
    
    # Initialize analyzer
    analyzer = FinfetAnalyzer(pixel_size=args.pixel_size)
    
    try:
        # Load image
        analyzer.load_image(file_path)
        
        # Detect fins
        analyzer.detect_fins(
            sigma=args.fin_sigma,
            thresh_method=args.threshold_method,
            min_size=args.min_size,
            max_regions=args.max_regions
        )
        
        # Measure fin widths
        fin_widths = analyzer.measure_fins()
        
        # Detect and measure coating
        try:
            analyzer.detect_coating(
                sigma=args.coating_sigma,
                edge_method=args.edge_method
            )
            coating_measurements = analyzer.measure_coating()
        except Exception as e:
            print(f"Warning: Could not analyze coating: {str(e)}")
            coating_measurements = None
        
        # Visualize results
        fig = analyzer.visualize_results()
        
        if args.show:
            plt.show()
        
        # Save results
        analyzer.save_results(output_dir=args.output)
        
        return fin_widths, coating_measurements
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process a single file
        process_single_file(args.input, args)
        
    elif os.path.isdir(args.input):
        # Process all image files in the directory
        files_processed = 0
        supported_formats = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        
        for filename in os.listdir(args.input):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
                file_path = os.path.join(args.input, filename)
                result = process_single_file(file_path, args)
                
                if result is not None:
                    files_processed += 1
        
        if files_processed == 0:
            print("No supported image files were found in the directory.")
    
    else:
        print(f"Error: Input path {args.input} does not exist")
        sys.exit(1)

if __name__ == "__main__":
    main() 