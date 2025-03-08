#!/usr/bin/env python3
"""Image handling module for TEM analysis with functions for loading and processing images."""

import numpy as np
from skimage import io, filters, exposure, transform, morphology, color
from skimage.feature import canny as canny_filter

class ImageHandler:
    """Handler for TEM image loading and preprocessing operations."""
    
    @staticmethod
    def load_dm3(file_path):
        """Load a Digital Micrograph (.dm3) TEM image file.
        
        Args:
            file_path: Path to the .dm3 file
            
        Returns:
            Tuple of (image_data, metadata, scale_factors)
        """
        try:
            from ncempy.io import dm
            # Load the DM file using ncempy
            dmFile = dm.fileDM(file_path)
            metadata = dmFile.allTags
            image_data = dmFile.getDataset(0)
            
            # Extract scale information (pixel size)
            scale_x = metadata.get('Pixel Size X', None)
            scale_y = metadata.get('Pixel Size Y', None)
            
            # If scale info found, convert to nm/pixel
            scale_factors = None
            if scale_x is not None and scale_y is not None:
                scale_factors = (scale_x, scale_y)
                
            return image_data['data'], metadata, scale_factors
            
        except ImportError:
            print("Warning: ncempy not installed. Cannot read .dm3 files properly.")
            try:
                # Fallback to using imageio
                import imageio
                img = imageio.imread(file_path)
                return img, None, None
            except:
                raise ValueError(f"Failed to load .dm3 file: {file_path}")
    
    @staticmethod
    def load_image(file_path):
        """Load a general image file (JPEG, PNG, TIFF, etc.).
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple of (image_data, metadata, scale_factors)
        """
        # Check if file is a .dm3 file
        if file_path.lower().endswith('.dm3'):
            return ImageHandler.load_dm3(file_path)
            
        # Otherwise load as a normal image file
        try:
            img = io.imread(file_path)
            
            # Convert RGB to grayscale if needed
            if len(img.shape) > 2 and img.shape[2] in [3, 4]:
                print(f"Converting RGB image of shape {img.shape} to grayscale")
                gray_img = color.rgb2gray(img)
                
                # Check if this is an SEM image with info bar at the bottom
                # Typical SEM images have an info bar in the bottom ~10% of the image
                height = gray_img.shape[0]
                info_bar_height = int(height * 0.1)  # Approximate height of info bar
                
                # Look for horizontal line that might separate image from info bar
                # by checking for sudden brightness changes in the lower part of the image
                lower_part = gray_img[height-int(height*0.2):height, :]
                row_means = np.mean(lower_part, axis=1)
                
                # Find significant changes in brightness
                row_diffs = np.abs(np.diff(row_means))
                threshold = np.mean(row_diffs) + 2 * np.std(row_diffs)
                significant_changes = np.where(row_diffs > threshold)[0]
                
                # If we found a significant change, use it to crop the image
                if len(significant_changes) > 0:
                    # Get the position of the most significant change
                    crop_point = height - int(height*0.2) + significant_changes[-1]
                    print(f"Detected SEM info bar, cropping image at row {crop_point}")
                    gray_img = gray_img[:crop_point, :]
                
                return gray_img, None, None
                
            return img, None, None
            
        except Exception as e:
            raise ValueError(f"Failed to load image file: {file_path}. Error: {str(e)}")
    
    @staticmethod
    def preprocess(image, sigma=1.0, denoise=False, normalize=True, clahe=False, **kwargs):
        """Preprocess an image with common operations for TEM analysis.
        
        Args:
            image: Input image array
            sigma: Sigma for Gaussian smoothing
            denoise: Whether to apply denoising
            normalize: Whether to normalize image values
            clahe: Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed image
        """
        if image is None:
            raise ValueError("Cannot preprocess None image")
            
        # Make a copy of the input image
        processed = image.copy().astype(np.float32)
        
        # Ensure image is 2D (grayscale)
        if len(processed.shape) > 2:
            print(f"Warning: Image has shape {processed.shape}, converting to grayscale")
            processed = color.rgb2gray(processed)
        
        # Rescale if image has values outside [0,1]
        if processed.max() > 1.0:
            processed = exposure.rescale_intensity(processed)
        
        # Apply denoising if requested
        if denoise:
            from skimage.restoration import denoise_nl_means
            processed = denoise_nl_means(processed, 
                                        patch_size=5, 
                                        patch_distance=6, 
                                        h=kwargs.get('h', 0.08))
        
        # Apply Gaussian smoothing
        if sigma > 0:
            processed = filters.gaussian(processed, sigma=sigma)
        
        # Normalize image
        if normalize:
            processed = exposure.rescale_intensity(processed)
        
        # Apply CLAHE if requested
        if clahe:
            processed = exposure.equalize_adapthist(processed, 
                                                   kernel_size=kwargs.get('clahe_kernel_size', 64),
                                                   clip_limit=kwargs.get('clahe_clip_limit', 0.01))
        
        return processed
    
    @staticmethod
    def detect_edges(image, method='sobel', threshold=None):
        """Detect edges in an image using various methods.
        
        Args:
            image: Input image array
            method: Edge detection method ('sobel', 'canny', 'scharr', 'prewitt')
            threshold: Threshold value for edge detection (if None, automatic threshold is used)
            
        Returns:
            Edge image
        """
        # Ensure image is 2D (grayscale)
        if len(image.shape) > 2:
            image = color.rgb2gray(image)
            
        if method == 'sobel':
            edges = filters.sobel(image)
        elif method == 'canny':
            # Use the imported canny from skimage.feature
            sigma = 1.0
            low_threshold = 0.1 if threshold is None else threshold*0.5
            high_threshold = 0.2 if threshold is None else threshold
            edges = canny_filter(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        elif method == 'scharr':
            edges = filters.scharr(image)
        elif method == 'prewitt':
            edges = filters.prewitt(image)
        else:
            raise ValueError(f"Unsupported edge detection method: {method}")
            
        return edges