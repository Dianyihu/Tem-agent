import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import HomeScreen from './pages/HomeScreen';
import CopilotMode from './pages/CopilotMode';
import AgentMode from './pages/AgentMode';

// Create a theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" className="app-container">
        <Router>
          <Routes>
            <Route path="/" element={<HomeScreen />} />
            <Route path="/copilot" element={<CopilotMode />} />
            <Route path="/agent" element={<AgentMode />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </Container>
    </ThemeProvider>
  );
}

export default App; 