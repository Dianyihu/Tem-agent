# TEM Analysis Tools for LLM Agents

A Python package providing TEM (Transmission Electron Microscopy) image analysis tools designed for use with LLM agents, focused on FinFET structure analysis.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tem-agent.git
cd tem-agent
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Architecture

The package is structured to provide modular TEM analysis capabilities through a unified interface:

```
tem_agent/
├── core/               # Core functionality
│   ├── base.py         # Base analyzer class
│   ├── image.py        # Image handling
│   └── metrics.py      # Measurement utilities
├── analyzers/          # Specialized analyzers  
│   └── finfet.py       # FinFET structure analyzer
└── utils/              # Utility functions
```

## Usage for LLM Agents

The package is designed to be easily used by LLM agents through a consistent interface:

```python
from tem_agent import create_analyzer

# Create the FinFET analyzer
analyzer = create_analyzer("finfet")

# Load an image
analyzer.load_image("finfet.jpeg")

# Run analysis with default parameters
results = analyzer.analyze()

# Or customize the analysis
results = analyzer.analyze(
    sigma=2.0,
    thresh_method="otsu",
    min_size=150,
    run_coating_analysis=True
)

# Get formatted results suitable for LLM consumption
report = results.get_summary()
print(report)

# Visualize results
analyzer.visualize()
```

## Available Analyzers

- **FinfetAnalyzer**: Analyzes FinFET structures, measuring fin width and coating thickness

## LLM Agent Integration

LLM agents can interact with the package through a standardized API:

1. **Task Selection**: Identify appropriate FinFET analyzer settings for the task
2. **Parameter Configuration**: Set analysis parameters based on image characteristics
3. **Result Interpretation**: Process and explain the analysis results
4. **Visualization**: Generate visual outputs for human inspection

## Example

To run the included example with the sample FinFET image:

```bash
python examples/llm_agent_example.py
```

This will run the analysis on the included finfet.jpeg sample image using default parameters. For custom options:

```bash
python examples/llm_agent_example.py --params '{"sigma": 2.5, "min_size": 200}'
```

# Film Thickness Analyzer

A tool for analyzing thin film structures and labeling the thickness of each layer in TEM (Transmission Electron Microscopy) images.

## Overview

This tool analyzes TEM images of thin films to:
1. Identify individual layers in the film structure
2. Measure the thickness of each layer
3. Create visualizations with labeled thickness values
4. Generate statistical summaries of layer measurements

## Requirements

- Python 3.6 or higher
- Dependencies from the TEM agent package:
  - NumPy
  - SciPy
  - Matplotlib
  - scikit-image
  - pandas (for the notebook only)

## Installation

1. Make sure you have the `tem_agent` package installed in your environment
2. Place the script files in a directory of your choice

## Usage

### Command Line Interface

You can run the analyzer from the command line:

```bash
python film_thickness_analyzer.py thin_fim.png --pixel-size 0.5 --method advanced --output-dir results
```

Arguments:
- `image_path`: Path to the thin film image (required)
- `--pixel-size`: Size in nm per pixel (optional, will try to extract from metadata if not provided)
- `--method`: Analysis method, either 'basic' or 'advanced' (default: 'advanced')
- `--output-dir`: Directory to save results (default: 'results')

### Jupyter Notebook

A Jupyter notebook `film_thickness_analysis.ipynb` is provided for interactive analysis. To use it:

1. Start Jupyter notebook or Jupyter lab
2. Open `film_thickness_analysis.ipynb`
3. Follow the steps in the notebook to analyze your thin film image

## Analysis Methods

The tool provides two methods for film layer analysis:

1. **Basic Method**: Uses edge detection and thresholding to identify layer boundaries.
   - Good for simple film structures with clear contrast between layers
   - Faster computation
   - May struggle with complex, multilayer films

2. **Advanced Method**: Uses watershed segmentation for more sophisticated layer detection.
   - Better for complex multilayer films
   - Can detect more subtle layer boundaries
   - May require more computational resources

## Output

The analyzer produces:
- Segmented image showing identified layers
- Thickness measurements for each layer
- Statistical summary (mean, standard deviation, min, max thickness)
- Visualization with thickness labels overlaid on the segmented image

## Example

When analyzing a thin film image, the output will include:
- A visualization saved to the output directory
- A printed summary of layer thickness measurements
- Detailed measurements for each identified layer

## Troubleshooting

If you encounter issues:
1. Check that the image file exists and is readable
2. Try adjusting the pixel size if measurements seem incorrect
3. Try both 'basic' and 'advanced' methods to see which works better for your image
4. Ensure the image has sufficient contrast between layers

## License

This tool is part of the TEM agent package and follows the same licensing terms. 