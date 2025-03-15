/**
 * Calculate the angle between a line and the horizontal axis
 */
export const calculateAngle = (startX: number, startY: number, endX: number, endY: number): number => {
  const deltaX = endX - startX;
  const deltaY = endY - startY;
  
  // Calculate angle in radians and convert to degrees
  let angleDegrees = Math.atan2(deltaY, deltaX) * (180 / Math.PI);
  
  // We want the negative of the calculated angle since we're determining 
  // the correction angle needed to make the line horizontal
  // (counter-clockwise is negative, clockwise is positive)
  let correctionAngle = -angleDegrees;
  
  // Normalize the angle to be between -90 and 90 degrees
  if (correctionAngle > 90) {
    correctionAngle -= 180;
  } else if (correctionAngle < -90) {
    correctionAngle += 180;
  }
  
  return parseFloat(correctionAngle.toFixed(2));
};

/**
 * Rotate an image by the given angle
 */
export const rotateImage = (
  imageElement: HTMLImageElement, 
  angleInDegrees: number
): Promise<string> => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (!ctx) {
      resolve('');
      return;
    }
    
    // Convert degrees to radians for the canvas rotation
    const angleInRadians = angleInDegrees * (Math.PI / 180);
    
    // Calculate new canvas dimensions to fit the rotated image
    const imgWidth = imageElement.width;
    const imgHeight = imageElement.height;
    
    // Get the rotated dimensions
    const rotatedWidth = Math.abs(imgWidth * Math.cos(angleInRadians)) + Math.abs(imgHeight * Math.sin(angleInRadians));
    const rotatedHeight = Math.abs(imgWidth * Math.sin(angleInRadians)) + Math.abs(imgHeight * Math.cos(angleInRadians));
    
    canvas.width = rotatedWidth;
    canvas.height = rotatedHeight;
    
    // Move to the center of the canvas
    ctx.translate(rotatedWidth / 2, rotatedHeight / 2);
    
    // Rotate the canvas
    ctx.rotate(angleInRadians);
    
    // Draw the image centered on the canvas
    ctx.drawImage(imageElement, -imgWidth / 2, -imgHeight / 2, imgWidth, imgHeight);
    
    // Get the data URL of the rotated image
    const dataUrl = canvas.toDataURL('image/png');
    resolve(dataUrl);
  });
};

/**
 * Load an image from a file
 */
export const loadImageFromFile = (file: File): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = e.target?.result as string;
    };
    
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

/**
 * Format an angle for display
 */
export const formatAngle = (angle: number): string => {
  const direction = angle >= 0 ? 'clockwise' : 'counter-clockwise';
  return `${Math.abs(angle)}° ${direction}`;
}; 