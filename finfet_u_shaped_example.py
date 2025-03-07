#!/usr/bin/env python3
"""
Example script for analyzing U-shaped FINFET structures in TEM images.
"""

import os
import sys
import traceback
print("Starting script...")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import matplotlib.pyplot as plt
    print("Matplotlib imported successfully")
    import numpy as np
    print("NumPy imported successfully")
    from skimage import measure
    print("Skimage measure imported successfully")
    from tem_agent import FinfetAnalyzer
    print("FinfetAnalyzer imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    """Run FINFET analysis on a U-shaped fin structure."""
    # Path to your image file (replace with your actual file path)
    image_file = "your_finfet_image.jpg"  # Replace with actual path
    
    # Check if file was provided as command line argument
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    
    print(f"Target image file: {image_file}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"File exists: {os.path.exists(image_file)}")
    
    # Check if the file exists
    if not os.path.exists(image_file):
        print(f"File {image_file} does not exist.")
        print("Please provide a path to your FINFET image as a command line argument.")
        print("Example: python finfet_u_shaped_example.py path/to/your/image.jpg")
        return
    
    # Create output directory
    output_dir = "finfet_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Extract scale factor from filename if available
    # For example, if filename contains "10nm" or similar
    scale_factor = None
    if "_" in image_file and "nm" in image_file:
        try:
            # Very simple attempt to extract scale information from filename
            parts = os.path.basename(image_file).split('_')
            for part in parts:
                if part.endswith('nm'):
                    scale_factor = float(part.replace('nm', ''))
                    break
        except:
            scale_factor = None
    
    # Try to extract scale from image if it contains a scale bar
    # For this specific image, we know it has a scale bar of 10 nm
    if "finfet.jpeg" in image_file.lower():
        print("Detected finfet.jpeg - applying known scale factor of 10nm")
        scale_factor = 10.0 / 100  # Approximate scale: 10nm scale bar is about 100 pixels
    
    print(f"Using scale factor: {scale_factor}")
    
    # Initialize analyzer specifically for U-shaped fins with automatic parameter optimization
    print(f"Analyzing U-shaped FINFET image: {image_file}")
    try:
        analyzer = FinfetAnalyzer(pixel_size=scale_factor, u_shaped=True)
        print("FinfetAnalyzer initialized successfully")
    except Exception as e:
        print(f"Error initializing FinfetAnalyzer: {e}")
        traceback.print_exc()
        return
    
    # Load the image
    try:
        print("Loading image...")
        analyzer.load_image(image_file)
        print("Image loaded successfully")
        # Print image shape to verify it's loaded properly
        print(f"Image shape: {analyzer.image.shape if hasattr(analyzer, 'image') and analyzer.image is not None else 'No image'}")
    except Exception as e:
        print(f"Error loading image: {e}")
        traceback.print_exc()
        return
    
    # For the finfet.jpeg image, use parameters that trigger specialized detection
    min_size = 1000  # Higher min_size to trigger specialized detection
    
    # Approach 1: Use specialized detection for U-shaped FINFET
    print("\nUsing specialized detection for U-shaped FINFET structures...")
    try:
        fins = analyzer.detect_fins(
            sigma=2.0,              
            thresh_method='otsu',   
            min_size=min_size,     # Use higher min_size to trigger specialized detection
            max_regions=5           
        )
        print(f"Fin detection result: {len(fins)} fins detected")
    except Exception as e:
        print(f"Error detecting fins: {e}")
        traceback.print_exc()
        fins = []
    
    if len(fins) > 0:
        print(f"Successfully detected {len(fins)} fins!")
        try:
            # Measure fins with more sampling profiles for better accuracy
            print("Measuring fin wall thickness...")
            fin_widths = analyzer.measure_fins(num_profiles=20)
            print(f"Measured {len(fin_widths)} fin widths")
            
            # Direct save to CSV for debugging
            try:
                with open(os.path.join(output_dir, "debug_fin_wall_thickness.csv"), 'w') as f:
                    f.write("fin_id,wall_thickness_px,wall_thickness_nm\n")
                    for i, (region, width, _) in enumerate(fin_widths):
                        scaled_width = width * scale_factor
                        f.write(f"{i+1},{width:.4f},{scaled_width:.4f}\n")
                print(f"Debug fin wall thickness saved to {output_dir}/debug_fin_wall_thickness.csv")
            except Exception as e:
                print(f"Error saving debug fin wall thickness: {e}")
            
            # Detect coating
            print("Detecting coating...")
            analyzer.detect_coating(sigma=1.5, edge_method='canny')
            
            # Measure coating
            print("Measuring coating...")
            analyzer.measure_coating()
            
            # Visualize
            print("Generating visualization...")
            fig = analyzer.visualize_results()
            
            # Save manually
            print(f"Saving figure to {output_dir}/finfet_analysis_result.png")
            plt.savefig(os.path.join(output_dir, "finfet_analysis_result.png"), dpi=300)
            
            # Save results
            print(f"Saving analysis results to {output_dir}")
            analyzer.save_results(output_dir=output_dir)
            
            # Save another copy of the figure for backup
            try:
                plt.figure(figsize=(12, 10))
                plt.imshow(analyzer.image, cmap='gray')
                plt.title("FINFET Image with Wall Thickness Measurement")
                
                # Add annotations from fin_widths
                for i, (region, width, profiles) in enumerate(fin_widths):
                    y0, x0 = region.centroid
                    minr, minc, maxr, maxc = region.bbox
                    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(x0, y0, f"Fin {i+1}", color='yellow', 
                            backgroundcolor='black', fontsize=12)
                    
                    # Draw a representative profile to show wall thickness
                    if profiles:
                        # Use a middle profile for visualization
                        rep_profile = profiles[len(profiles)//2]
                        (pos_y, pos_x), (dir_y, dir_x), thickness = rep_profile
                        
                        # Draw a line perpendicular to the wall
                        scale = 20
                        start_y = pos_y - dir_y * scale
                        start_x = pos_x - dir_x * scale
                        end_y = pos_y + dir_y * scale
                        end_x = pos_x + dir_x * scale
                        
                        plt.plot([start_x, end_x], [start_y, end_y], 'g-', linewidth=2)
                        plt.text(pos_x + 10, pos_y, f"Wall: {width*scale_factor:.2f} nm", 
                                color='white', backgroundcolor='green', fontsize=10)
                
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "wall_thickness_result.png"), dpi=300)
                print(f"Wall thickness visualization saved to {output_dir}/wall_thickness_result.png")
            except Exception as e:
                print(f"Error saving wall thickness visualization: {e}")
                traceback.print_exc()
                
            print("Analysis completed and results saved")
        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
    else:
        print("No fins detected with specialized parameters.")
        
        # Fallback to manual region extraction
        print("\nFallback: Manually defining regions based on image dimensions...")
        try:
            # Analyze image dimensions
            height, width = analyzer.image.shape[:2]
            
            # Create a dummy mask for the fins based on manual coordinates
            mask = np.zeros((height, width), dtype=bool)
            
            # Define approximate regions for the two U-shaped fins
            # Left fin
            left_x_start = int(width * 0.25)
            left_x_end = int(width * 0.45)
            # Right fin
            right_x_start = int(width * 0.55)
            right_x_end = int(width * 0.75)
            # Common Y range
            y_start = int(height * 0.15)
            y_end = int(height * 0.75)
            
            # Set the regions in the mask
            mask[y_start:y_end, left_x_start:left_x_end] = True
            mask[y_start:y_end, right_x_start:right_x_end] = True
            
            # Label the regions
            labeled = measure.label(mask)
            regions = measure.regionprops(labeled)
            
            print(f"Manual region definition created {len(regions)} fin regions")
            
            # Save visualization of manually defined regions
            plt.figure(figsize=(10, 8))
            plt.imshow(analyzer.image, cmap='gray')
            plt.title("Original Image with Manual Region Overlay")
            
            for i, region in enumerate(regions):
                minr, minc, maxr, maxc = region.bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
                y, x = region.centroid
                plt.text(x, y, f"{i+1}", color='yellow',
                        backgroundcolor='black', fontsize=12)
                
                # Width measurement
                width = maxc - minc
                plt.text(x, y + 30, f"Width: {width:.2f} px", 
                        color='white', backgroundcolor='green', fontsize=10)
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "manual_regions.png"))
            print(f"Manual region analysis saved to {output_dir}/manual_regions.png")
        except Exception as e:
            print(f"Error during manual region definition: {e}")
            traceback.print_exc()
            
            # Save the raw image for reference
            plt.figure(figsize=(10, 8))
            plt.imshow(analyzer.image, cmap='gray')
            plt.title("Original Image (No Fins Detected)")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, "original_image.png"))
            print("Saved original image for reference")
    
    print(f"\nResults saved to {output_dir}")
    
    # Don't show plots to avoid blocking
    # plt.show()

if __name__ == "__main__":
    try:
        print("Starting main function")
        main()
        print("Main function completed")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc() 