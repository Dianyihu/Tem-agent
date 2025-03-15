import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Line, Image as KonvaImage } from 'react-konva';
import { Box } from '@mui/material';

interface DrawingCanvasProps {
  imageUrl: string;
  onLineDrawn: (startX: number, startY: number, endX: number, endY: number) => void;
  width: number;
  height: number;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({ imageUrl, onLineDrawn, width, height }) => {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [lines, setLines] = useState<Array<number[]>>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [scale, setScale] = useState(1);
  
  const stageRef = useRef<any>(null);
  
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
  }, [imageUrl, width, height]);

  const handleMouseDown = () => {
    setIsDrawing(true);
    // Clear previous lines
    setLines([]);
    
    // Get pointer position
    const stage = stageRef.current;
    const pointerPos = stage.getPointerPosition();
    
    setLines([[pointerPos.x, pointerPos.y]]);
  };

  const handleMouseMove = () => {
    if (!isDrawing) return;
    
    const stage = stageRef.current;
    const pointerPos = stage.getPointerPosition();
    
    // We only support a single line for angle measurement
    if (lines.length === 1) {
      const updatedLine = [...lines[0], pointerPos.x, pointerPos.y];
      setLines([updatedLine]);
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    
    if (lines.length === 1 && lines[0].length === 4) {
      const [startX, startY, endX, endY] = lines[0];
      
      // Scale the coordinates back to match the original image dimensions
      const originalStartX = startX / scale;
      const originalStartY = startY / scale;
      const originalEndX = endX / scale;
      const originalEndY = endY / scale;
      
      onLineDrawn(originalStartX, originalStartY, originalEndX, originalEndY);
    }
  };

  return (
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
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line}
              stroke="#ff0000"
              strokeWidth={2}
              tension={0}
              lineCap="round"
              lineJoin="round"
            />
          ))}
        </Layer>
      </Stage>
    </Box>
  );
};

export default DrawingCanvas; 