# TEM Image Tilt Detector

A web application for batch processing TEM (Transmission Electron Microscopy) images to calculate and correct tilt angles.

## Features

- **Two Operation Modes**:
  - **Copilot Mode**: Interactive processing where users manually draw reference lines on images to calculate tilt angles.
  - **Agent Mode**: Automated processing where the system detects tilt angles for all images in a folder.

- **Key Capabilities**:
  - Select folders containing multiple images
  - Draw reference lines to indicate the desired horizontal plane
  - Calculate precise tilt angles
  - Rotate images to correct the tilt
  - Save processed images with corrected orientation
  - Generate reports with tilt angle data

## Getting Started

### Prerequisites

- Node.js (>= 14.x)
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tem-tilt-detector
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

### Copilot Mode

1. Click on "Enter Copilot Mode" from the home screen
2. Select a folder containing TEM images
3. For each image:
   - Draw a line along what should be horizontal
   - The app will calculate the tilt angle
   - Click "Rotate Image" to level the image
   - Save the rotated image or move to the next one

### Agent Mode

1. Click on "Enter Agent Mode" from the home screen
2. Select a folder containing TEM images
3. Click "Start Processing"
4. Once processing is complete, review the results
5. Save the results to a text file for documentation

## Supported File Formats

The application supports the following image formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Technical Details

This application is built using:
- React with TypeScript
- Material UI for the interface components
- Konva for canvas drawing functionality
- HTML5 Canvas API for image processing

## License

This project is licensed under the MIT License - see the LICENSE file for details. 