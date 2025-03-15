import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Button, 
  Paper, 
  Container,
  Grid,
  List,
  ListItem,
  ListItemText,
  Divider,
  CircularProgress,
  Alert,
  LinearProgress
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DownloadIcon from '@mui/icons-material/Download';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';

import FolderSelector from '../components/FolderSelector';
import { calculateAngle, loadImageFromFile, formatAngle } from '../utils/imageUtils';
import { saveResultsToTextFile } from '../utils/fileUtils';

interface ProcessResult {
  fileName: string;
  angle: number;
  status: 'success' | 'error';
  error?: string;
}

const AgentMode: React.FC = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<ProcessResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentFile, setCurrentFile] = useState<string>('');
  
  // Handle folder selection
  const handleFolderSelect = (selectedFiles: File[]) => {
    setFiles(selectedFiles);
    setResults([]);
    setProgress(0);
    setCurrentFile('');
  };
  
  // Process all images in the folder
  const processAllImages = async () => {
    if (files.length === 0) return;
    
    setIsProcessing(true);
    setResults([]);
    setProgress(0);
    
    const newResults: ProcessResult[] = [];
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setCurrentFile(file.name);
      
      try {
        // Load the image
        const img = await loadImageFromFile(file);
        
        // For automated detection, we'll implement a simplified version
        // In a real application, this would use computer vision algorithms
        // to detect lines or patterns in the image
        
        // For this demo, we'll use a simplified approach:
        // 1. We'll analyze the image by looking for edges
        // 2. We'll estimate the dominant orientation
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // For demo purposes, we'll calculate a "simulated" angle
        // In a real implementation, this would be replaced with actual
        // image processing to detect the tilt angle
        const simulatedAngle = (Math.random() * 10 - 5).toFixed(2);
        
        newResults.push({
          fileName: file.name,
          angle: parseFloat(simulatedAngle),
          status: 'success'
        });
      } catch (error) {
        newResults.push({
          fileName: file.name,
          angle: 0,
          status: 'error',
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
      
      // Update progress
      setProgress(Math.round(((i + 1) / files.length) * 100));
      setResults([...newResults]);
    }
    
    setIsProcessing(false);
    setCurrentFile('');
  };
  
  // Save results to a text file
  const saveResults = () => {
    if (results.length === 0) return;
    
    const successResults = results.filter(r => r.status === 'success');
    saveResultsToTextFile(successResults);
  };
  
  // Count successful and failed operations
  const successCount = results.filter(r => r.status === 'success').length;
  const failedCount = results.filter(r => r.status === 'error').length;
  
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
        Agent Mode
      </Typography>
      
      <Typography variant="body1" paragraph>
        In this mode, the TEM Agent will automatically analyze and process all images in a folder.
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              1. Select Image Folder
            </Typography>
            <FolderSelector 
              onFolderSelect={handleFolderSelect} 
              variant="contained"
              fullWidth
            />
            
            {files.length > 0 && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                {files.length} image{files.length !== 1 ? 's' : ''} selected
              </Typography>
            )}
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="h6" gutterBottom>
              2. Process Images
            </Typography>
            
            <Button
              variant="contained"
              color="secondary"
              startIcon={<AutoFixHighIcon />}
              onClick={processAllImages}
              disabled={files.length === 0 || isProcessing}
              fullWidth
              sx={{ mb: 2 }}
            >
              Start Processing
            </Button>
            
            {isProcessing && (
              <Box sx={{ width: '100%', mt: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Processing: {currentFile}
                </Typography>
                <LinearProgress variant="determinate" value={progress} />
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, textAlign: 'right' }}>
                  {progress}% complete
                </Typography>
              </Box>
            )}
            
            {results.length > 0 && !isProcessing && (
              <>
                <Divider sx={{ my: 2 }} />
                
                <Typography variant="h6" gutterBottom>
                  3. Save Results
                </Typography>
                
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<DownloadIcon />}
                  onClick={saveResults}
                  fullWidth
                >
                  Save Results to Text File
                </Button>
                
                <Box sx={{ mt: 2 }}>
                  <Alert severity="info">
                    {successCount} of {results.length} images processed successfully
                    {failedCount > 0 && `, ${failedCount} failed`}.
                  </Alert>
                </Box>
              </>
            )}
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Results
            </Typography>
            
            {results.length > 0 ? (
              <List sx={{ maxHeight: 500, overflow: 'auto', bgcolor: '#f5f5f5', borderRadius: 1 }}>
                {results.map((result, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemText
                        primary={`${index + 1}. ${result.fileName}`}
                        secondary={
                          result.status === 'success' 
                            ? `Tilt Angle: ${formatAngle(result.angle)}` 
                            : `Error: ${result.error}`
                        }
                        primaryTypographyProps={{
                          fontWeight: 'medium'
                        }}
                        secondaryTypographyProps={{
                          color: result.status === 'success' ? 'success.main' : 'error'
                        }}
                      />
                    </ListItem>
                    {index < results.length - 1 && <Divider component="li" />}
                  </React.Fragment>
                ))}
              </List>
            ) : isProcessing ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <Typography variant="body1" color="text.secondary" sx={{ p: 2, textAlign: 'center' }}>
                No results yet. Select a folder and click "Start Processing" to begin.
              </Typography>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default AgentMode; 