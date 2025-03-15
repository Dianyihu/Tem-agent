import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Line, Image as KonvaImage } from 'react-konva';
import { Box, Typography } from '@mui/material';
import Konva from 'konva';

interface DrawingCanvasProps {
  imageUrl: string;
  onLineDrawn: (startX: number, startY: number, endX: number, endY: number) => void;
  width: number;
  height: number;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({ imageUrl, onLineDrawn, width, height }) => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [startPoint, setStartPoint] = useState<{x: number, y: number} | null>(null);
  const [endPoint, setEndPoint] = useState<{x: number, y: number} | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [scale, setScale] = useState(1);
  
  const stageRef = useRef<Konva.Stage | null>(null);
  
  // Load the image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.src = imageUrl;
    img.onload = () => {
      setImage(img);
      
      // Calculate scale to fit the image within the canvas
      const scaleX = width / img.width;
      const scaleY = height / img.height;
      const newScale = Math.min(scaleX, scaleY);
      setScale(newScale);
    };
    
    // Reset points when image changes
    setStartPoint(null);
    setEndPoint(null);
  }, [imageUrl, width, height]);

  const handleMouseDown = () => {
    setIsDrawing(true);
    
    // Clear previous points
    setStartPoint(null);
    setEndPoint(null);
    
    // Get pointer position
    const stage = stageRef.current;
    if (!stage) return;
    
    const pointerPos = stage.getPointerPosition();
    if (!pointerPos) return;
    
    // Set start point
    setStartPoint({x: pointerPos.x, y: pointerPos.y});
  };

  const handleMouseMove = () => {
    if (!isDrawing || !startPoint) return;
    
    const stage = stageRef.current;
    if (!stage) return;
    
    const pointerPos = stage.getPointerPosition();
    if (!pointerPos) return;
    
    // Update end point as mouse moves
    setEndPoint({x: pointerPos.x, y: pointerPos.y});
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    
    if (startPoint && endPoint) {
      // Make sure start and end points are different enough to calculate angle
      const dx = endPoint.x - startPoint.x;
      const dy = endPoint.y - startPoint.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // Only trigger onLineDrawn if the line is long enough (at least 10 pixels)
      if (distance >= 10) {
        // Scale the coordinates back to match the original image dimensions
        const originalStartX = startPoint.x / scale;
        const originalStartY = startPoint.y / scale;
        const originalEndX = endPoint.x / scale;
        const originalEndY = endPoint.y / scale;
        
        onLineDrawn(originalStartX, originalStartY, originalEndX, originalEndY);
      } else {
        // Line too short - reset points
        console.log("Line too short, please draw a longer line");
      }
    }
  };

  // Get the line points for drawing
  const getLinePoints = () => {
    if (!startPoint || !endPoint) return [];
    return [startPoint.x, startPoint.y, endPoint.x, endPoint.y];
  };
  
  // Get instructions message
  const getInstructions = () => {
    if (!startPoint) return "Click and drag to draw a reference line";
    if (startPoint && !endPoint) return "Now drag to complete the line";
    return "";
  };

  return (
    <Box sx={{ position: 'relative' }}>
      <Box 
        sx={{ 
          width: width, 
          height: height, 
          border: '1px solid #ccc',
          overflow: 'hidden',
          cursor: 'crosshair' 
        }}
      >
        <Stage
          width={width}
          height={height}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onTouchStart={handleMouseDown}
          onTouchMove={handleMouseMove}
          onTouchEnd={handleMouseUp}
          ref={stageRef}
        >
          <Layer>
            {image && (
              <KonvaImage
                image={image}
                width={image.width * scale}
                height={image.height * scale}
                x={(width - image.width * scale) / 2}
                y={(height - image.height * scale) / 2}
              />
            )}
            {startPoint && endPoint && (
              <Line
                points={getLinePoints()}
                stroke="#ff0000"
                strokeWidth={2}
                tension={0}
                lineCap="round"
                lineJoin="round"
              />
            )}
          </Layer>
        </Stage>
      </Box>
      <Typography 
        variant="caption" 
        sx={{ 
          position: 'absolute', 
          bottom: 8, 
          left: 8, 
          background: 'rgba(255,255,255,0.7)', 
          padding: '2px 8px', 
          borderRadius: 1 
        }}
      >
        {getInstructions()}
      </Typography>
    </Box>
  );
};

export default DrawingCanvas; 