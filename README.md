# TEM Image Analysis Tool

A Python tool for automatically analyzing TEM (.dm3) images and measuring thin film thickness on Si substrates.

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

## Usage

### Basic Usage

```bash
python tem_analyzer.py --input path/to/your/image.dm3 --output results/
```

### Options

- `--input`: Path to input .dm3 file or directory containing .dm3 files
- `--output`: Directory to save analysis results
- `--threshold`: Threshold value for edge detection (default: 0.5)
- `--show`: Display visualizations during analysis
- `--save-plots`: Save visualization plots to output directory

## Features

- Automatic loading and processing of .dm3 TEM image files
- Detection of thin film-substrate interface
- Measurement of thin film thickness
- Statistical analysis of thickness variation
- Visualization of results

## Example

```python
from tem_agent import TemAnalyzer

# Initialize analyzer
analyzer = TemAnalyzer()

# Load image
analyzer.load_image("sample.dm3")

# Analyze film thickness
results = analyzer.measure_thickness()

# Display results
analyzer.visualize_results()

# Get thickness statistics
mean_thickness, std_dev = analyzer.get_thickness_stats()
print(f"Mean thickness: {mean_thickness:.2f} nm ± {std_dev:.2f} nm")
``` 