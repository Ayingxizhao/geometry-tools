"""
Line detection utilities using Hough Transform.
Detects lines in edge-detected images and extracts their properties.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def detect_lines_hough(edges: np.ndarray,
                       rho: float = 1,
                       theta: float = np.pi/180,
                       threshold: int = 50,
                       min_line_length: int = 30,
                       max_line_gap: int = 10) -> Optional[np.ndarray]:
    """
    Detect lines using Probabilistic Hough Transform.
    
    Args:
        edges: Edge-detected image (from Canny)
        rho: Distance resolution in pixels
        theta: Angle resolution in radians
        threshold: Minimum number of votes to consider a line
        min_line_length: Minimum line length in pixels
        max_line_gap: Maximum gap between line segments to treat as single line
        
    Returns:
        Array of detected lines, shape (N, 1, 4) where each line is [x1, y1, x2, y2]
        Returns None if no lines detected
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    return lines


def calculate_line_length(line: np.ndarray) -> float:
    """
    Calculate Euclidean length of a line segment.
    
    Args:
        line: Line coordinates [x1, y1, x2, y2]
        
    Returns:
        Length in pixels
    """
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculate_line_angle(line: np.ndarray) -> float:
    """
    Calculate angle of line relative to horizontal axis.
    
    Args:
        line: Line coordinates [x1, y1, x2, y2]
        
    Returns:
        Angle in degrees (0-180)
    """
    x1, y1, x2, y2 = line
    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
    return angle


def get_line_properties(line: np.ndarray) -> Dict:
    """
    Extract all properties of a line segment.
    
    Args:
        line: Line coordinates [x1, y1, x2, y2]
        
    Returns:
        Dictionary with line properties
    """
    x1, y1, x2, y2 = line
    
    return {
        'endpoints': [(x1, y1), (x2, y2)],
        'length': calculate_line_length(line),
        'angle': calculate_line_angle(line),
        'midpoint': ((x1 + x2) / 2, (y1 + y2) / 2),
        'coordinates': [x1, y1, x2, y2]
    }


def filter_horizontal_lines(lines: np.ndarray, 
                           angle_tolerance: float = 15.0) -> List[np.ndarray]:
    """
    Filter to keep only approximately horizontal lines.
    Useful for Müller-Lyer illusion where main shafts are horizontal.
    
    Args:
        lines: Array of detected lines
        angle_tolerance: Maximum deviation from horizontal (in degrees)
        
    Returns:
        List of horizontal lines
    """
    if lines is None:
        return []
    
    horizontal_lines = []
    
    for line in lines:
        coords = line[0]
        angle = calculate_line_angle(coords)
        
        # Check if close to 0° or 180° (horizontal)
        if angle < angle_tolerance or angle > (180 - angle_tolerance):
            horizontal_lines.append(coords)
    
    return horizontal_lines


def merge_similar_lines(lines: List[np.ndarray],
                       distance_threshold: float = 10.0,
                       angle_threshold: float = 5.0) -> List[np.ndarray]:
    """
    Merge lines that are very close and parallel.
    Helps reduce duplicate detections.
    
    Args:
        lines: List of line coordinates
        distance_threshold: Maximum distance between midpoints to merge
        angle_threshold: Maximum angle difference to merge
        
    Returns:
        List of merged lines
    """
    if not lines:
        return []
    
    merged = []
    used = set()
    
    for i, line1 in enumerate(lines):
        if i in used:
            continue
            
        # Start a group with this line
        group = [line1]
        used.add(i)
        
        prop1 = get_line_properties(line1)
        
        for j, line2 in enumerate(lines[i+1:], start=i+1):
            if j in used:
                continue
                
            prop2 = get_line_properties(line2)
            
            # Check if similar
            angle_diff = abs(prop1['angle'] - prop2['angle'])
            midpoint_dist = np.sqrt(
                (prop1['midpoint'][0] - prop2['midpoint'][0])**2 +
                (prop1['midpoint'][1] - prop2['midpoint'][1])**2
            )
            
            if angle_diff < angle_threshold and midpoint_dist < distance_threshold:
                group.append(line2)
                used.add(j)
        
        # Merge group into single line (use longest)
        longest = max(group, key=calculate_line_length)
        merged.append(longest)
    
    return merged


def detect_and_filter_lines(edges: np.ndarray,
                           filter_horizontal: bool = True,
                           merge_duplicates: bool = True) -> List[Dict]:
    """
    Complete pipeline: detect lines and return cleaned results with properties.
    
    Args:
        edges: Edge-detected image
        filter_horizontal: Whether to keep only horizontal lines
        merge_duplicates: Whether to merge similar lines
        
    Returns:
        List of dictionaries, each containing line properties
    """
    # Detect lines
    raw_lines = detect_lines_hough(edges)
    
    if raw_lines is None:
        return []
    
    # Extract coordinates
    lines = [line[0] for line in raw_lines]
    
    # Filter horizontal if requested
    if filter_horizontal:
        lines = filter_horizontal_lines(raw_lines)
    
    # Merge duplicates if requested
    if merge_duplicates and lines:
        lines = merge_similar_lines(lines)
    
    # Get properties for each line
    results = []
    for i, line in enumerate(lines):
        props = get_line_properties(line)
        props['id'] = f"line_{i}"
        results.append(props)
    
    return results