import React, { useRef } from 'react';
import { Button, Box, Typography } from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import { isImageFile } from '../utils/fileUtils';

interface FolderSelectorProps {
  onFolderSelect: (files: File[]) => void;
  variant?: 'contained' | 'outlined';
  color?: 'primary' | 'secondary';
  fullWidth?: boolean;
}

const FolderSelector: React.FC<FolderSelectorProps> = ({ 
  onFolderSelect, 
  variant = 'contained', 
  color = 'primary',
  fullWidth = false
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFolderSelect = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { files } = event.target;
    
    if (files && files.length > 0) {
      // Convert FileList to Array and filter for images
      const fileArray = Array.from(files).filter(isImageFile);
      
      if (fileArray.length === 0) {
        alert('No valid image files found in the selected folder.');
        return;
      }
      
      // Sort files by name for consistent processing
      fileArray.sort((a, b) => a.name.localeCompare(b.name));
      
      onFolderSelect(fileArray);
    }
  };

  return (
    <Box>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        webkitdirectory="true"
        directory=""
        multiple
        onChange={handleFileChange}
      />
      <Button
        variant={variant}
        color={color}
        startIcon={<FolderIcon />}
        onClick={handleFolderSelect}
        fullWidth={fullWidth}
      >
        Select Image Folder
      </Button>
      <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary' }}>
        Supported formats: JPEG, PNG, GIF, BMP, TIFF
      </Typography>
    </Box>
  );
};

export default FolderSelector; 