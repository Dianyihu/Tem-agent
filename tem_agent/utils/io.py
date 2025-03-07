import os
import numpy as np
import matplotlib.pyplot as plt
from hyperspy.io import load
import cv2

def load_dm3(file_path):
    """Load a .dm3 file and return its data as a numpy array."""
    try:
        signal = load(file_path)
        # Extract the data as a numpy array
        data = signal.data
        # Get metadata if available
        metadata = signal.metadata
        
        scale_factors = None
        try:
            # Try to extract scale information (nm/pixel)
            scale_x = signal.axes_manager[0].scale
            scale_y = signal.axes_manager[1].scale
            scale_factors = (scale_x, scale_y)
        except:
            scale_factors = None
            
        return data, metadata, scale_factors
    except Exception as e:
        print(f"Error loading DM3 file: {str(e)}")
        return None, None, None

def load_image(file_path):
    """Load an image file (JPEG, PNG, etc.) and return its data as a numpy array."""
    try:
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.dm3':
            return load_dm3(file_path)
        
        # For normal image formats
        img = cv2.imread(file_path)
        
        if img is None:
            raise ValueError(f"Could not load image: {file_path}")
            
        # Convert BGR to RGB for better compatibility with matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # No metadata or scale factors for regular images
        return img_rgb, None, None
    except Exception as e:
        print(f"Error loading image file: {str(e)}")
        return None, None, None

def save_results(output_dir, filename, data, measurements=None, mean_thickness=None, std_thickness=None):
    """Save analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Base filename without extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Save results as numpy array
    if data is not None:
        np.save(os.path.join(output_dir, f"{base_name}_data.npy"), data)
    
    # Save measurements as CSV
    if measurements:
        with open(os.path.join(output_dir, f"{base_name}_measurements.csv"), 'w') as f:
            f.write("x,y,thickness_nm\n")
            for (y, x), thickness in measurements:
                f.write(f"{x},{y},{thickness:.4f}\n")
    
    # Save summary as text file
    if mean_thickness is not None:
        with open(os.path.join(output_dir, f"{base_name}_summary.txt"), 'w') as f:
            f.write(f"Mean thickness: {mean_thickness:.4f} nm\n")
            if std_thickness is not None:
                f.write(f"Standard deviation: {std_thickness:.4f} nm\n")
                f.write(f"Relative standard deviation: {(std_thickness/mean_thickness*100):.2f}%\n")

def save_finfet_results(output_dir, filename, fin_widths, coating_measurements=None):
    """Save FINFET analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Base filename without extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Save fin width measurements as CSV
    with open(os.path.join(output_dir, f"{base_name}_fin_widths.csv"), 'w') as f:
        f.write("fin_id,centroid_y,centroid_x,width_px,major_axis_length,minor_axis_length,orientation\n")
        for i, (region, width) in enumerate(fin_widths):
            y, x = region.centroid
            f.write(f"{i+1},{y:.2f},{x:.2f},{width:.4f},{region.major_axis_length:.4f},"
                    f"{region.minor_axis_length:.4f},{region.orientation:.4f}\n")
    
    # Save coating measurements if available
    if coating_measurements:
        with open(os.path.join(output_dir, f"{base_name}_coating_thickness.csv"), 'w') as f:
            f.write("fin_id,avg_thickness_px,std_thickness_px\n")
            for i, coating in enumerate(coating_measurements):
                avg = coating['avg_thickness']
                std = coating['std_thickness']
                f.write(f"{i+1},{avg:.4f},{std:.4f}\n")
        
        # Save detailed measurements
        with open(os.path.join(output_dir, f"{base_name}_coating_detailed.csv"), 'w') as f:
            f.write("fin_id,point_y,point_x,thickness_px\n")
            for i, coating in enumerate(coating_measurements):
                for (y, x), thickness in coating['points']:
                    f.write(f"{i+1},{y},{x},{thickness:.4f}\n")
    
    # Save summary as text file
    with open(os.path.join(output_dir, f"{base_name}_finfet_summary.txt"), 'w') as f:
        f.write(f"Number of fins detected: {len(fin_widths)}\n")
        for i, (region, width) in enumerate(fin_widths):
            f.write(f"Fin {i+1} width: {width:.4f} px\n")
        
        if coating_measurements:
            f.write("\nCoating thickness:\n")
            for i, coating in enumerate(coating_measurements):
                avg = coating['avg_thickness']
                std = coating['std_thickness']
                f.write(f"Fin {i+1} coating: {avg:.4f} ± {std:.4f} px\n")

def save_plot(fig, output_dir, filename, dpi=300):
    """Save a matplotlib figure to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Base filename without extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Save as PNG and PDF
    fig.savefig(os.path.join(output_dir, f"{base_name}_plot.png"), dpi=dpi, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{base_name}_plot.pdf"), bbox_inches='tight')

def batch_process_directory(input_dir, output_dir, process_func, file_extension='.dm3'):
    """Process all files with the given extension in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(file_extension):
            file_path = os.path.join(input_dir, filename)
            print(f"Processing: {file_path}")
            
            # Process the file
            result = process_func(file_path, output_dir)
            results.append((filename, result))
    
    return results 