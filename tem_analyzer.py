#!/usr/bin/env python3
"""
TEM Image Analyzer - Command line tool for analyzing TEM images and measuring thin film thickness.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
from tem_agent import TemAnalyzer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze TEM images and measure thin film thickness."
    )
    
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to input .dm3 file or directory containing .dm3 files"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="results",
        help="Directory to save analysis results (default: results)"
    )
    
    parser.add_argument(
        "--pixel-size", "-p", 
        type=float,
        help="Pixel size in nm/pixel (overrides metadata if provided)"
    )
    
    parser.add_argument(
        "--sigma", "-s", 
        type=float, 
        default=1.0,
        help="Sigma value for Gaussian smoothing (default: 1.0)"
    )
    
    parser.add_argument(
        "--edge-method", "-e", 
        choices=["sobel", "canny", "scharr"], 
        default="sobel",
        help="Edge detection method (default: sobel)"
    )
    
    parser.add_argument(
        "--threshold-method", "-t", 
        choices=["otsu", "yen", "mean"], 
        default="otsu",
        help="Thresholding method (default: otsu)"
    )
    
    parser.add_argument(
        "--min-region-size", "-m", 
        type=int, 
        default=100,
        help="Minimum region size in pixels (default: 100)"
    )
    
    parser.add_argument(
        "--advanced", "-a", 
        action="store_true",
        help="Use advanced layer detection algorithm"
    )
    
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show visualization plots during analysis"
    )
    
    parser.add_argument(
        "--save-plots", 
        action="store_true",
        help="Save visualization plots to output directory"
    )
    
    return parser.parse_args()

def process_single_file(file_path, args):
    """Process a single TEM image file."""
    print(f"Processing file: {file_path}")
    
    # Initialize analyzer
    analyzer = TemAnalyzer(pixel_size=args.pixel_size)
    
    try:
        # Load image
        analyzer.load_image(file_path)
        
        # Preprocess
        analyzer.preprocess(sigma=args.sigma)
        
        # Detect interface
        if args.advanced:
            analyzer.advanced_layer_detection(min_size=args.min_region_size)
        else:
            analyzer.detect_film_substrate_interface(
                edge_method=args.edge_method,
                threshold_method=args.threshold_method,
                min_region_size=args.min_region_size
            )
        
        # Measure thickness
        measurements, mean_thickness, std_thickness = analyzer.measure_thickness()
        
        # Visualize results
        fig = analyzer.visualize_results()
        
        if args.show:
            plt.show()
        
        # Save results
        analyzer.save_results(output_dir=args.output)
        
        return mean_thickness, std_thickness
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None, None

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        if not args.input.lower().endswith('.dm3'):
            print(f"Error: Input file {args.input} is not a .dm3 file")
            sys.exit(1)
        
        # Process a single file
        process_single_file(args.input, args)
        
    elif os.path.isdir(args.input):
        # Process all .dm3 files in the directory
        files_processed = 0
        results = []
        
        for filename in os.listdir(args.input):
            if filename.lower().endswith('.dm3'):
                file_path = os.path.join(args.input, filename)
                mean, std = process_single_file(file_path, args)
                
                if mean is not None:
                    results.append((filename, mean, std))
                    files_processed += 1
        
        # Print summary of all processed files
        if files_processed > 0:
            print("\nSummary of all processed files:")
            print("-" * 60)
            print(f"{'Filename':<30} {'Mean (nm)':<15} {'Std Dev (nm)':<15}")
            print("-" * 60)
            
            for filename, mean, std in results:
                print(f"{filename:<30} {mean:<15.2f} {std:<15.2f}")
            
            # Calculate overall statistics
            means = [mean for _, mean, _ in results]
            overall_mean = sum(means) / len(means)
            print("-" * 60)
            print(f"Overall mean thickness: {overall_mean:.2f} nm")
        else:
            print("No .dm3 files were found in the directory.")
    
    else:
        print(f"Error: Input path {args.input} does not exist")
        sys.exit(1)

if __name__ == "__main__":
    main() 