#!/usr/bin/env python3
"""Base analyzer class that provides common functionality for all TEM analysis tasks."""

import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """Abstract base class for all TEM image analyzers."""
    
    def __init__(self, pixel_size=None):
        """Initialize base analyzer with common attributes.
        
        Args:
            pixel_size: Physical size per pixel in nm/pixel, if known
        """
        self.image = None
        self.metadata = None
        self.processed_image = None
        self.filename = None
        self.pixel_size = pixel_size
        self.scale_factor = 1.0  # Will be updated if calibration info is available
        self.results = None
        
    @abstractmethod
    def load_image(self, file_path):
        """Load an image from a file (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def preprocess(self, **kwargs):
        """Preprocess the loaded image (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def analyze(self, **kwargs):
        """Perform analysis on the image (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def visualize(self, **kwargs):
        """Visualize analysis results (to be implemented by subclasses)."""
        pass
    
    @abstractmethod
    def save_results(self, output_dir="results", **kwargs):
        """Save analysis results to files (to be implemented by subclasses)."""
        pass
    
    def get_image_info(self):
        """Get basic information about the loaded image."""
        if self.image is None:
            return "No image loaded"
        
        info = {
            "filename": self.filename,
            "shape": self.image.shape,
            "dtype": str(self.image.dtype),
            "scale_factor": f"{self.scale_factor} nm/pixel" if self.scale_factor != 1.0 else "unknown"
        }
        
        # Add extracted metadata if available
        if hasattr(self, 'metadata') and self.metadata:
            for key, value in self.metadata.items():
                if key not in info and key != 'allTags':  # Avoid duplicates and large tag collections
                    info[key] = value
        
        return info
    
    def _ensure_output_dir(self, output_dir):
        """Ensure the output directory exists."""
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def run_pipeline(self, preprocess_kwargs=None, analysis_kwargs=None, save_dir=None):
        """Run the complete analysis pipeline with a single call.
        
        Args:
            preprocess_kwargs: Dictionary of preprocessing parameters
            analysis_kwargs: Dictionary of analysis parameters
            save_dir: Directory to save results
            
        Returns:
            Analysis results
        """
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        # Use empty dictionaries as defaults if None
        preprocess_kwargs = preprocess_kwargs or {}
        analysis_kwargs = analysis_kwargs or {}
            
        # Preprocess
        self.preprocess(**preprocess_kwargs)
        
        # Analyze
        results = self.analyze(**analysis_kwargs)
        
        # Save results if a directory is provided
        if save_dir:
            self.save_results(output_dir=save_dir)
            
        return results

class AnalysisResult:
    """Class to store and format analysis results."""
    
    def __init__(self, data, metadata=None):
        """Initialize with analysis data and metadata.
        
        Args:
            data: The analysis results data (measurements, regions, etc.)
            metadata: Additional metadata about the analysis
        """
        self.data = data
        self.metadata = metadata or {}
        
    def get_summary(self, format="text"):
        """Get a formatted summary of the results.
        
        Args:
            format: Output format ('text', 'json', 'html')
            
        Returns:
            Formatted summary of results
        """
        if format == "text":
            return self._get_text_summary()
        elif format == "json":
            return self._get_json_summary()
        elif format == "html":
            return self._get_html_summary()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _get_text_summary(self):
        """Generate a text summary of the results."""
        # Default implementation - override in subclasses
        return str(self.data)
    
    def _get_json_summary(self):
        """Generate a JSON summary of the results."""
        # Default implementation - override in subclasses
        import json
        return json.dumps(self.data)
    
    def _get_html_summary(self):
        """Generate an HTML summary of the results."""
        # Default implementation - override in subclasses
        html = "<h2>Analysis Results</h2>"
        html += f"<pre>{str(self.data)}</pre>"
        return html