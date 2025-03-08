#!/usr/bin/env python3
"""Example script demonstrating how an LLM agent would use the TEM analyzer package for FinFET analysis."""

import os
import json
import sys
import argparse

# Add the parent directory to the path so we can import tem_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tem_agent import create_analyzer

def simulate_llm_agent(analyzer_type, image_path, analysis_parameters=None, output_dir="results"):
    """Simulate an LLM agent using the TEM analysis tools.
    
    Args:
        analyzer_type: Type of analyzer to create ('finfet' or 'fin')
        image_path: Path to the image to analyze
        analysis_parameters: Optional parameters for the analysis
        output_dir: Directory to save results
    """
    # Step 1: Print agent thinking process (simulating LLM agent internal processing)
    print("=" * 80)
    print("LLM Agent Processing")
    print("=" * 80)
    print(f"Task: Analyze {analyzer_type} structure in TEM image")
    print(f"Input image: {image_path}")
    print(f"Analysis parameters: {analysis_parameters}")
    print("-" * 80)
    print("Determining appropriate analyzer type...")
    print(f"Selected analyzer: {analyzer_type}")
    print("-" * 80)
    
    # Step 2: Create the appropriate analyzer (what an LLM would do via API)
    print("Creating analyzer...")
    analyzer = create_analyzer(analyzer_type)
    
    # Step 3: Load the image
    print(f"Loading image from {image_path}...")
    analyzer.load_image(image_path)
    
    # Step 4: Run the analysis with parameters determined by LLM
    print("Running analysis with the following parameters:")
    if analysis_parameters:
        for param, value in analysis_parameters.items():
            print(f"  - {param}: {value}")
        results = analyzer.analyze(**analysis_parameters)
    else:
        print("  Using default parameters")
        results = analyzer.analyze()
    
    # Step 5: Get a summary of the results
    print("-" * 80)
    print("Analysis Results Summary:")
    print("-" * 80)
    print(results.get_summary())
    
    # Step 6: Generate visualization (optional)
    print("Generating visualization...")
    analyzer.visualize(show=False)
    
    # Step 7: Save the results
    print(f"Saving results to {output_dir}...")
    analyzer.save_results(output_dir=output_dir)
    
    # Step 8: LLM agent provides a conclusion
    print("=" * 80)
    print("LLM Agent Conclusion")
    print("=" * 80)
    
    # For finfet analyzer
    width_stats = results.metadata.get('width_statistics', {})
    coating_stats = results.metadata.get('coating_statistics', {})
    
    print(f"I've analyzed the FinFET structures in the TEM image {os.path.basename(image_path)}.")
    print(f"Detected {width_stats.get('count', 0)} fin structures with an average width of {width_stats.get('mean', 0):.2f} nm.")
    
    if coating_stats.get('count', 0) > 0:
        print(f"The fins have a coating with average thickness of {coating_stats.get('mean', 0):.2f} nm.")
        
    print(f"Analysis results and visualization have been saved to {output_dir}.")
    
    return results

def main():
    """Parse arguments and run the example."""
    parser = argparse.ArgumentParser(description='Simulate LLM agent using TEM analyzers')
    
    parser.add_argument('--analyzer', '-a', default='finfet',
                        choices=['finfet', 'fin'],
                        help='Type of analyzer to use (default: finfet)')
                        
    parser.add_argument('--image', '-i', default='finfet.jpeg',
                        help='Path to the image file to analyze (default: finfet.jpeg)')
                        
    parser.add_argument('--params', '-p',
                        help='JSON string of parameters to pass to the analyzer')
                        
    parser.add_argument('--output', '-o', default='results',
                        help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Parse parameters if provided
    analysis_parameters = None
    if args.params:
        try:
            analysis_parameters = json.loads(args.params)
        except json.JSONDecodeError:
            print(f"Error: Could not parse parameters JSON: {args.params}")
            return
    
    # Run the simulation
    simulate_llm_agent(
        analyzer_type=args.analyzer,
        image_path=args.image,
        analysis_parameters=analysis_parameters,
        output_dir=args.output
    )

if __name__ == "__main__":
    main() 