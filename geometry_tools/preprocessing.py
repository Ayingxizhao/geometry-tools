"""
Image preprocessing utilities for geometry detection.
Handles loading, grayscale conversion, and edge detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image in BGR format (OpenCV default)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    return image


def preprocess_for_line_detection(image: np.ndarray, 
                                   blur_kernel: int = 5,
                                   canny_low: int = 50,
                                   canny_high: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for line detection using Canny edge detection.
    
    Args:
        image: Input image (BGR or grayscale)
        blur_kernel: Gaussian blur kernel size (must be odd)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        
    Returns:
        Tuple of (grayscale_image, edge_detected_image)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    return gray, edges


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about the image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image properties
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'height': height,
        'width': width,
        'channels': channels,
        'shape': image.shape,
        'dtype': str(image.dtype)
    }