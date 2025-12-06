"""
Shape detection and size comparison algorithms for vision_2d_Compare_Size task.
Supports circles, squares, and rectangles with size-based comparisons.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def detect_shapes_by_contours(edges: np.ndarray, min_area: float = 500.0, max_area: float = 50000.0) -> List[np.ndarray]:
    """
    Detect shapes using contour analysis.
    
    Args:
        edges: Edge-detected image (binary)
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
    
    Returns:
        List of contour arrays representing detected shapes
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            valid_contours.append(contour)
    
    logger.info(f"Found {len(valid_contours)} valid shapes (area range: {min_area}-{max_area})")
    return valid_contours

def classify_shape_type(contour: np.ndarray, circularity_threshold: float = 0.85) -> Dict[str, Any]:
    """
    Classify a contour as circle, square, or rectangle based on geometric properties.
    
    Args:
        contour: OpenCV contour
        circularity_threshold: Minimum circularity to be considered a circle
    
    Returns:
        Dictionary with shape classification and properties
    """
    # Calculate basic properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return {
            'type': 'unknown',
            'area': area,
            'perimeter': perimeter,
            'circularity': 0,
            'vertices': 0
        }
    
    # Calculate circularity (4π * area / perimeter²)
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Approximate polygon to count vertices
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # Classify based on properties
    if circularity >= circularity_threshold:
        shape_type = 'circle'
    elif vertices == 4:
        # Check if it's a square (aspect ratio close to 1)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        if 0.8 <= aspect_ratio <= 1.2:  # Allow some tolerance for squares
            shape_type = 'square'
        else:
            shape_type = 'rectangle'
    else:
        shape_type = 'other'
    
    return {
        'type': shape_type,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'vertices': vertices,
        'centroid': calculate_centroid(contour)
    }

def calculate_centroid(contour: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the centroid (center point) of a contour.
    
    Args:
        contour: OpenCV contour
    
    Returns:
        Tuple of (x, y) coordinates
    """
    if len(contour) == 0:
        return (0.0, 0.0)
    
    M = cv2.moments(contour)
    if M['m00'] == 0:
        # Fallback to bounding box center
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w/2, y + h/2)
    
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    return (cx, cy)

def detect_and_classify_shapes(image: np.ndarray, min_area: float = 500.0, max_area: float = 50000.0) -> Dict[str, Any]:
    """
    Detect all shapes in an image and classify them by type.
    
    Args:
        image: Input image (RGB or BGR)
        min_area: Minimum shape area to consider
        max_area: Maximum shape area to consider
    
    Returns:
        Dictionary with detection results
    """
    try:
        # Convert to grayscale if needed (image is RGB from load_image)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours = detect_shapes_by_contours(edges, min_area, max_area)
        
        if len(contours) == 0:
            return {
                'success': False,
                'error': 'No shapes detected in image',
                'num_shapes': 0,
                'shapes': []
            }
        
        # Classify each shape
        shapes = []
        for i, contour in enumerate(contours):
            shape_info = classify_shape_type(contour)
            shape_info['id'] = f'shape_{i}'
            shape_info['contour'] = contour
            shapes.append(shape_info)
        
        # Sort shapes by x-coordinate (left to right)
        shapes.sort(key=lambda s: s['centroid'][0])
        
        return {
            'success': True,
            'num_shapes': len(shapes),
            'shapes': shapes,
            'image_size': image.shape[:2]
        }
        
    except Exception as e:
        logger.error(f"Error in shape detection: {e}")
        return {
            'success': False,
            'error': str(e),
            'num_shapes': 0,
            'shapes': []
        }

def compare_shape_sizes(shapes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare sizes of detected shapes (shape 1 vs shape 2).
    
    Args:
        shapes: List of shape dictionaries from detect_and_classify_shapes
    
    Returns:
        Dictionary with comparison results
    """
    if len(shapes) < 2:
        return {
            'success': False,
            'error': f'Need at least 2 shapes for comparison, found {len(shapes)}'
        }
    
    # Get first two shapes (leftmost = shape_1, next = shape_2)
    shape_1 = shapes[0]
    shape_2 = shapes[1]
    
    # Compare areas
    area_1 = shape_1['area']
    area_2 = shape_2['area']
    
    is_larger = area_1 > area_2
    area_ratio = area_1 / area_2 if area_2 > 0 else float('inf')
    
    return {
        'success': True,
        'shape_1': {
            'id': shape_1['id'],
            'type': shape_1['type'],
            'area': area_1,
            'centroid': shape_1['centroid']
        },
        'shape_2': {
            'id': shape_2['id'],
            'type': shape_2['type'],
            'area': area_2,
            'centroid': shape_2['centroid']
        },
        'is_shape_1_larger': is_larger,
        'area_ratio': area_ratio,
        'area_difference': area_1 - area_2
    }

def validate_shape_comparison(image: np.ndarray, expected_larger: bool, min_area: float = 500.0, max_area: float = 50000.0) -> Dict[str, Any]:
    """
    Validate if shape 1 is larger than shape 2 according to expectation.
    
    Args:
        image: Input image
        expected_larger: Expected result (True if shape 1 should be larger)
        min_area: Minimum shape area to consider
        max_area: Maximum shape area to consider
    
    Returns:
        Dictionary with validation results
    """
    # Detect shapes
    detection_result = detect_and_classify_shapes(image, min_area, max_area)
    
    if not detection_result['success']:
        return {
            'success': False,
            'error': detection_result['error'],
            'is_valid': False
        }
    
    # Compare sizes
    comparison_result = compare_shape_sizes(detection_result['shapes'])
    
    if not comparison_result['success']:
        return {
            'success': False,
            'error': comparison_result['error'],
            'is_valid': False
        }
    
    # Validate against expectation
    is_valid = comparison_result['is_shape_1_larger'] == expected_larger
    
    return {
        'success': True,
        'is_valid': is_valid,
        'shape_1_larger': comparison_result['is_shape_1_larger'],
        'expected_larger': expected_larger,
        'area_ratio': comparison_result['area_ratio'],
        'shapes': detection_result['shapes'],
        'comparison': comparison_result
    }
