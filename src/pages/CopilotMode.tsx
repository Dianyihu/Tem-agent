import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Container,
  Grid,
  Divider,
  Alert,
  CircularProgress
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CropRotateIcon from '@mui/icons-material/CropRotate';
import SaveIcon from '@mui/icons-material/Save';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import InfoIcon from '@mui/icons-material/Info';

import FolderSelector from '../components/FolderSelector';
import DrawingCanvas from '../components/DrawingCanvas';
import { calculateAngle, rotateImage, formatAngle } from '../utils/imageUtils';
import { saveImageFile } from '../utils/fileUtils';

const CopilotMode: React.FC = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [currentImageUrl, setCurrentImageUrl] = useState<string>('');
  const [rotatedImageUrl, setRotatedImageUrl] = useState<string>('');
  const [tiltAngle, setTiltAngle] = useState<number | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lineDrawn, setLineDrawn] = useState(false);
  
  const canvasWidth = 600;
  const canvasHeight = 400;
  
  // Handle folder selection
  const handleFolderSelect = (selectedFiles: File[]) => {
    setFiles(selectedFiles);
    setCurrentFileIndex(0);
    loadImage(selectedFiles[0]);
    
    // Reset state for new image set
    setTiltAngle(null);
    setRotatedImageUrl('');
    setLineDrawn(false);
  };
  
  // Load image file into view
  const loadImage = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        setCurrentImageUrl(e.target.result as string);
        setRotatedImageUrl('');
        setTiltAngle(null);
        setLineDrawn(false);
      }
    };
    reader.readAsDataURL(file);
  };
  
  // Handle line drawing on image
  const handleLineDrawn = (startX: number, startY: number, endX: number, endY: number) => {
    const angle = calculateAngle(startX, startY, endX, endY);
    setTiltAngle(angle);
    setLineDrawn(true);
  };
  
  // Process image rotation
  const handleRotateImage = async () => {
    if (tiltAngle === null || !files[currentFileIndex]) return;
    
    setIsProcessing(true);
    
    try {
      // Create an image element from the current URL
      const img = new Image();
      img.src = currentImageUrl;
      
      await new Promise((resolve) => {
        img.onload = resolve;
      });
      
      // Rotate the image by the calculated angle
      const rotatedUrl = await rotateImage(img, tiltAngle);
      setRotatedImageUrl(rotatedUrl);
    } catch (error) {
      console.error('Error rotating image:', error);
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Save the processed image
  const handleSaveImage = () => {
    if (rotatedImageUrl && files[currentFileIndex]) {
      saveImageFile(rotatedImageUrl, files[currentFileIndex].name);
    }
  };
  
  // Move to next image
  const handleNextImage = () => {
    if (currentFileIndex < files.length - 1) {
      setCurrentFileIndex(currentFileIndex + 1);
      loadImage(files[currentFileIndex + 1]);
    }
  };
  
  // Check if there are any images loaded
  const hasImages = files.length > 0;
  
  // Check if this is the last image
  const isLastImage = currentFileIndex === files.length - 1;
  
  return (
    <Container>
      <Button 
        startIcon={<ArrowBackIcon />} 
        onClick={() => navigate('/')}
        sx={{ mt: 2, mb: 2 }}
      >
        Back to Home
      </Button>
      
      <Typography variant="h4" component="h1" gutterBottom>
        Copilot Mode
      </Typography>
      
      <Typography variant="body1" paragraph>
        In this mode, you can manually draw a reference line on each image to calculate the tilt angle.
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <FolderSelector onFolderSelect={handleFolderSelect} />
      </Box>
      
      {hasImages ? (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Image {currentFileIndex + 1} of {files.length}: {files[currentFileIndex]?.name}
              </Typography>
              
              <Box sx={{ position: 'relative' }}>
                {/* Show original image with drawing capabilities if no rotation yet */}
                {!rotatedImageUrl ? (
                  <DrawingCanvas 
                    imageUrl={currentImageUrl}
                    onLineDrawn={handleLineDrawn}
                    width={canvasWidth}
                    height={canvasHeight}
                  />
                ) : (
                  // Show rotated image
                  <Box 
                    sx={{ 
                      width: canvasWidth, 
                      height: canvasHeight, 
                      border: '1px solid #ccc',
                      overflow: 'hidden',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    <img 
                      src={rotatedImageUrl} 
                      alt="Rotated" 
                      style={{ maxWidth: '100%', maxHeight: '100%' }} 
                    />
                  </Box>
                )}
                
                {isProcessing && (
                  <Box 
                    sx={{ 
                      position: 'absolute', 
                      top: 0, 
                      left: 0, 
                      right: 0, 
                      bottom: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: 'rgba(255,255,255,0.7)'
                    }}
                  >
                    <CircularProgress />
                  </Box>
                )}
              </Box>
              
              <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {!rotatedImageUrl ? (
                  <Button 
                    variant="contained" 
                    color="primary"
                    startIcon={<CropRotateIcon />}
                    onClick={handleRotateImage}
                    disabled={!lineDrawn || isProcessing}
                  >
                    Rotate Image
                  </Button>
                ) : (
                  <>
                    <Button 
                      variant="contained" 
                      color="primary"
                      startIcon={<SaveIcon />}
                      onClick={handleSaveImage}
                    >
                      Save Rotated Image
                    </Button>
                    
                    <Button 
                      variant="outlined" 
                      color="primary"
                      onClick={() => setRotatedImageUrl('')}
                    >
                      Return to Original
                    </Button>
                  </>
                )}
                
                <Button 
                  variant="contained" 
                  color="secondary"
                  startIcon={<SkipNextIcon />}
                  onClick={handleNextImage}
                  disabled={isLastImage}
                >
                  Next Image
                </Button>
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Paper elevation={3} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Tilt Information
              </Typography>
              
              <Divider sx={{ mb: 2 }} />
              
              {lineDrawn ? (
                <Box>
                  <Typography variant="body1" gutterBottom>
                    <strong>Detected Tilt Angle:</strong> {tiltAngle !== null ? formatAngle(tiltAngle) : 'N/A'}
                  </Typography>
                  
                  {rotatedImageUrl && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Image has been rotated successfully!
                    </Alert>
                  )}
                </Box>
              ) : (
                <Alert severity="info" icon={<InfoIcon />}>
                  Draw a line on the image to indicate the horizontal reference.
                </Alert>
              )}
              
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Instructions:</strong>
                </Typography>
                <ol>
                  <li>Draw a line on the image that should be horizontal</li>
                  <li>The app will calculate the tilt angle</li>
                  <li>Click "Rotate Image" to level the image</li>
                  <li>Save the rotated image or go to the next one</li>
                </ol>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      ) : (
        <Paper elevation={3} sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>
            No Images Loaded
          </Typography>
          <Typography variant="body1">
            Please select a folder containing images to start processing.
          </Typography>
        </Paper>
      )}
    </Container>
  );
};

export default CopilotMode; 