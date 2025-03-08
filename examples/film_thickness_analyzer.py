#!/usr/bin/env python3
"""Script to analyze film structure and label thickness of each layer."""

import os
import sys
import argparse
import warnings

# Import the decoupled functions from thickness.py
from tem_agent.utils.thin_film import (
    load_image, 
    remove_sem_info_bar, 
    preprocess_image, 
    advanced_layer_detection, 
    measure_thickness, 
    create_visualization,
    analyze_film
)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze film structure and layer thickness")
    parser.add_argument("image_path", help="Path to the film image (supports .dm3, .tif, .png, etc.)")
    parser.add_argument("--pixel-size", type=float, help="Pixel size in nm/pixel (overrides metadata)")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of layers to detect")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--install-ncempy", action="store_true", 
                        help="Install ncempy for .dm3 file support and exit")
    
    args = parser.parse_args()
    
    # Handle the ncempy installation option
    if args.install_ncempy:
        try:
            import subprocess
            print("Installing ncempy for .dm3 file support...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ncempy"])
            print("ncempy installed successfully! You can now process .dm3 files.")
            return
        except Exception as e:
            print(f"Error installing ncempy: {str(e)}")
            print("Please install manually with: pip install ncempy")
            return
    
    # Run analysis using the decoupled function
    results = analyze_film(
        args.image_path,
        pixel_size=args.pixel_size,
        num_layers=args.num_layers,
        output_dir=args.output_dir
    )
    
    # Print summary
    print("\nResults Summary:")
    print(f"Analyzed {results['statistics']['count']} layers")
    print(f"Mean thickness: {results['statistics']['mean']:.2f} nm")
    print(f"Results saved to {results['output_path']}")
    print(f"Summary saved to {results['summary_path']}")

if __name__ == "__main__":
    main()