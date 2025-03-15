import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Typography, Button, Paper, Grid, Box } from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';

const HomeScreen: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box sx={{ marginTop: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        TEM Image Tilt Detector
      </Typography>
      
      <Typography variant="h6" component="h2" gutterBottom align="center" sx={{ mb: 4 }}>
        Process images and detect the angle they should tilt for leveling
      </Typography>

      <Grid container spacing={4} justifyContent="center">
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 3, 
              textAlign: 'center',
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'transform 0.3s',
              '&:hover': {
                transform: 'scale(1.02)',
              }
            }}
          >
            <Box sx={{ mb: 2 }}>
              <PersonIcon sx={{ fontSize: 60, color: 'primary.main' }} />
            </Box>
            <Typography variant="h5" component="h2" gutterBottom>
              Copilot Mode
            </Typography>
            <Typography variant="body1" sx={{ flexGrow: 1, mb: 3 }}>
              Interactive mode where you can manually draw a line on images to calculate tilt angles.
              Process images one by one with visual feedback.
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              size="large"
              onClick={() => navigate('/copilot')}
              fullWidth
            >
              Enter Copilot Mode
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 3, 
              textAlign: 'center',
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              transition: 'transform 0.3s',
              '&:hover': {
                transform: 'scale(1.02)',
              }
            }}
          >
            <Box sx={{ mb: 2 }}>
              <SmartToyIcon sx={{ fontSize: 60, color: 'secondary.main' }} />
            </Box>
            <Typography variant="h5" component="h2" gutterBottom>
              Agent Mode
            </Typography>
            <Typography variant="body1" sx={{ flexGrow: 1, mb: 3 }}>
              Automated mode where the system detects tilt angles for all images in a folder.
              Batch process images and get a report of the results.
            </Typography>
            <Button 
              variant="contained" 
              color="secondary" 
              size="large"
              onClick={() => navigate('/agent')}
              fullWidth
            >
              Enter Agent Mode
            </Button>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HomeScreen; 