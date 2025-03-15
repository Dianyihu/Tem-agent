import React, { useState, useEffect, useCallback } from 'react';
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
  CircularProgress,
  ThemeProvider,
  createTheme,
  alpha
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CropRotateIcon from '@mui/icons-material/CropRotate';
import SaveIcon from '@mui/icons-material/Save';
import SkipNextIcon from '@mui/icons-material/SkipNext';
import SkipPreviousIcon from '@mui/icons-material/SkipPrevious';
import InfoIcon from '@mui/icons-material/Info';
import RestoreIcon from '@mui/icons-material/Restore';

import FolderSelector from '../components/FolderSelector';
import DrawingCanvas from '../components/DrawingCanvas';
import { calculateAngle, rotateImage, formatAngle } from '../utils/imageUtils';
import { saveImageFile } from '../utils/fileUtils';

// Custom theme enhancement for buttons and typography
const theme = createTheme({
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
      fontSize: '1.75rem',
    },
    h6: {
      fontWeight: 600,
      fontSize: '1.1rem',
      letterSpacing: '0.02em',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.9rem',
      lineHeight: 1.5,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 16px',
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '0.95rem',
          boxShadow: '0 2px 4px rgba(0,0,0,0.08)',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 4px 8px rgba(0,0,0,0.12)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(45deg, #3f51b5 30%, #536dfe 90%)',
        },
        containedSecondary: {
          background: 'linear-gradient(45deg, #757de8 30%, #7986cb 90%)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
});

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
  
  // Process image rotation - defined with useCallback before it's used in useEffect
  const handleRotateImage = useCallback(async () => {
    if (tiltAngle === null || !files[currentFileIndex]) {
      console.log("Cannot rotate: tiltAngle or file is null", { tiltAngle, fileIndex: currentFileIndex });
      return;
    }
    
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
  }, [tiltAngle, files, currentFileIndex, currentImageUrl]);
  
  // Debug state changes
  useEffect(() => {
    console.log("Line drawn state:", lineDrawn);
    console.log("Tilt angle:", tiltAngle);
  }, [lineDrawn, tiltAngle]);
  
  // Auto-rotate when line is drawn
  useEffect(() => {
    if (lineDrawn && tiltAngle !== null && !rotatedImageUrl && !isProcessing) {
      handleRotateImage();
    }
  }, [lineDrawn, tiltAngle, rotatedImageUrl, isProcessing, handleRotateImage]);
  
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
    // Calculate angle based on drawn line coordinates
    const angle = calculateAngle(startX, startY, endX, endY);
    console.log("Line drawn with coordinates:", { startX, startY, endX, endY });
    console.log("Calculated angle:", angle);
    
    // Update state with angle and line drawn status
    setTiltAngle(angle);
    setLineDrawn(true);
    
    // Auto-rotation is handled by the useEffect
  };
  
  // Save the processed image
  const handleSaveImage = () => {
    if (rotatedImageUrl && files[currentFileIndex]) {
      saveImageFile(rotatedImageUrl, files[currentFileIndex].name);
    }
  };
  
  // Move to previous image
  const handlePreviousImage = () => {
    if (currentFileIndex > 0) {
      setCurrentFileIndex(currentFileIndex - 1);
      loadImage(files[currentFileIndex - 1]);
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
  
  // Check if this is the first or last image
  const isFirstImage = currentFileIndex === 0;
  const isLastImage = currentFileIndex === files.length - 1;
  
  // Determine if rotate button should be enabled
  const canRotate = lineDrawn && tiltAngle !== null && !isProcessing;
  
  // Return to original button click handler (defined with useCallback)
  const handleReturnToOriginal = useCallback(() => {
    setRotatedImageUrl('');
    setLineDrawn(false); // Reset the lineDrawn state so users can draw again
  }, []);
  
  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Button 
          startIcon={<ArrowBackIcon />} 
          onClick={() => navigate('/')}
          variant="outlined"
          size="large"
          sx={{ mb: 3 }}
        >
          Back to Home
        </Button>
        
        <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 2 }}>
          Copilot Mode
        </Typography>
        
        <Typography variant="body1" paragraph sx={{ mb: 3, maxWidth: '800px' }}>
          In this mode, you can manually draw a reference line on each image to calculate the tilt angle.
        </Typography>
        
        <Box sx={{ mb: 4 }}>
          <FolderSelector onFolderSelect={handleFolderSelect} />
        </Box>
        
        {hasImages ? (
          <Grid container spacing={4}>
            <Grid item xs={12} md={8}>
              <Paper elevation={3} sx={{ p: 3, overflow: 'hidden' }}>
                <Typography variant="h6" gutterBottom sx={{ mb: 2, color: '#3f51b5' }}>
                  Image {currentFileIndex + 1} of {files.length}: {files[currentFileIndex]?.name}
                </Typography>
                
                <Box 
                  sx={{ 
                    position: 'relative',
                    borderRadius: 2,
                    overflow: 'hidden',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                    mb: 3
                  }}
                >
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
                        overflow: 'hidden',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: alpha('#000', 0.03)
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
                        backgroundColor: 'rgba(255,255,255,0.7)',
                        zIndex: 2
                      }}
                    >
                      <CircularProgress size={60} />
                    </Box>
                  )}
                </Box>
                
                {/* Improved button layout with consistent spacing and alignment */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  {/* Action buttons (left side) */}
                  <Box>
                    {rotatedImageUrl ? (
                      <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button 
                          variant="contained" 
                          color="primary"
                          startIcon={<SaveIcon />}
                          onClick={handleSaveImage}
                          size="large"
                          sx={{ minWidth: '190px', height: '48px' }}
                        >
                          Save Rotated Image
                        </Button>
                        
                        <Button 
                          variant="outlined" 
                          color="primary"
                          startIcon={<RestoreIcon />}
                          onClick={handleReturnToOriginal}
                          size="large"
                          sx={{ minWidth: '170px', height: '48px' }}
                        >
                          Return to Original
                        </Button>
                      </Box>
                    ) : (
                      <Box>
                        {/* This is hidden since we auto-rotate, but keeping it in the code in case it's needed later */}
                        <Button 
                          variant="contained" 
                          color="primary"
                          startIcon={<CropRotateIcon />}
                          onClick={handleRotateImage}
                          disabled={!canRotate}
                          sx={{ display: 'none' }}
                        >
                          Rotate Image
                        </Button>
                      </Box>
                    )}
                  </Box>
                  
                  {/* Navigation buttons (right side) */}
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button 
                      variant="contained" 
                      color="secondary"
                      startIcon={<SkipPreviousIcon />}
                      onClick={handlePreviousImage}
                      disabled={isFirstImage}
                      size="large"
                      sx={{ 
                        minWidth: '170px', 
                        height: '48px',
                        opacity: isFirstImage ? 0.7 : 1
                      }}
                    >
                      Previous Image
                    </Button>
                    
                    <Button 
                      variant="contained" 
                      color="secondary"
                      startIcon={<SkipNextIcon />}
                      onClick={handleNextImage}
                      disabled={isLastImage}
                      size="large"
                      sx={{ 
                        minWidth: '170px', 
                        height: '48px',
                        opacity: isLastImage ? 0.7 : 1
                      }}
                    >
                      Next Image
                    </Button>
                  </Box>
                </Box>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom sx={{ mb: 2, color: '#3f51b5' }}>
                  Tilt Information
                </Typography>
                
                <Divider sx={{ mb: 3 }} />
                
                {lineDrawn ? (
                  <Box>
                    <Typography variant="body1" sx={{ fontSize: '1.1rem', mb: 1 }}>
                      <strong>Detected Tilt Angle:</strong>
                    </Typography>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        fontSize: '1.5rem', 
                        fontWeight: 600, 
                        color: '#333',
                        mb: 2,
                        pl: 2,
                        borderLeft: '4px solid #3f51b5'
                      }}
                    >
                      {tiltAngle !== null ? formatAngle(tiltAngle) : 'N/A'}
                    </Typography>
                    
                    {rotatedImageUrl && (
                      <Alert 
                        severity="success" 
                        sx={{ 
                          mt: 2, 
                          fontSize: '1rem',
                          py: 1.5,
                          backgroundColor: alpha('#4caf50', 0.1)
                        }}
                        icon={<Box component="span" sx={{ 
                          width: 24, 
                          height: 24, 
                          borderRadius: '50%', 
                          bgcolor: '#4caf50', 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center',
                          color: 'white',
                          fontWeight: 'bold',
                          fontSize: '16px'
                        }}>✓</Box>}
                      >
                        Image has been rotated successfully!
                      </Alert>
                    )}
                  </Box>
                ) : (
                  <Alert 
                    severity="info" 
                    icon={<InfoIcon sx={{ fontSize: '1.3rem' }} />}
                    sx={{ 
                      mb: 2, 
                      py: 1.5,
                      fontSize: '1rem',
                      backgroundColor: alpha('#2196f3', 0.1)
                    }}
                  >
                    Draw a straight line on the image that should be horizontal.
                  </Alert>
                )}
                
                <Box sx={{ mt: 4 }}>
                  <Typography 
                    variant="body2" 
                    color="text.secondary"
                    sx={{ 
                      fontSize: '1rem', 
                      fontWeight: 500, 
                      mb: 1.5,
                      color: '#555'
                    }}
                  >
                    <strong>Instructions:</strong>
                  </Typography>
                  <ol style={{ 
                    paddingLeft: '1.5rem', 
                    margin: 0
                  }}>
                    <li style={{ marginBottom: '0.7rem' }}>Draw a straight line on the image that should be horizontal</li>
                    <li style={{ marginBottom: '0.7rem' }}>The app will calculate the tilt angle and rotate automatically</li>
                    <li style={{ marginBottom: '0.7rem' }}>Save the rotated image or navigate to another image</li>
                  </ol>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        ) : (
          <Paper 
            elevation={3} 
            sx={{ 
              p: 4, 
              textAlign: 'center',
              maxWidth: '700px',
              mx: 'auto',
              borderRadius: 3
            }}
          >
            <Typography variant="h6" gutterBottom sx={{ mb: 2, color: '#3f51b5' }}>
              No Images Loaded
            </Typography>
            <Typography variant="body1" sx={{ fontSize: '1.1rem' }}>
              Please select a folder containing images to start processing.
            </Typography>
          </Paper>
        )}
      </Container>
    </ThemeProvider>
  );
};

export default CopilotMode; 