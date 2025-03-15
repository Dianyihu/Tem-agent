import { saveAs } from 'file-saver';

/**
 * Filter for allowed image file types
 */
export const isImageFile = (file: File): boolean => {
  const acceptedImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'];
  return acceptedImageTypes.includes(file.type);
};

/**
 * Save a dataURL as an image file
 */
export const saveImageFile = (dataUrl: string, fileName: string): void => {
  // Convert data URL to blob
  const parts = dataUrl.split(';base64,');
  const contentType = parts[0].split(':')[1];
  const raw = window.atob(parts[1]);
  const rawLength = raw.length;
  const uInt8Array = new Uint8Array(rawLength);

  for (let i = 0; i < rawLength; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }

  const blob = new Blob([uInt8Array], { type: contentType });
  
  // Add "_leveled" suffix to the original file name
  const fileNameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.'));
  const extension = fileName.substring(fileName.lastIndexOf('.'));
  const newFileName = `${fileNameWithoutExt}_leveled${extension}`;
  
  // Save the file
  saveAs(blob, newFileName);
};

/**
 * Save tilt angle results to a text file
 */
export const saveResultsToTextFile = (results: Array<{ fileName: string; angle: number }>): void => {
  let content = "TEM Image Tilt Analysis Results\n";
  content += "================================\n\n";
  
  results.forEach((result, index) => {
    content += `${index + 1}. ${result.fileName}\n`;
    content += `   Tilt Angle: ${result.angle}°\n\n`;
  });
  
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  saveAs(blob, 'tilt_angle_results.txt');
}; 