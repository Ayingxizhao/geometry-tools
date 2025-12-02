"""
Line detection utilities using Hough Transform.
Detects lines in edge-detected images and extracts their properties.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def detect_lines_hough(edges: np.ndarray,
                       rho: float = 2,
                       theta: float = np.pi/180,
                       threshold: int = 30,
                       min_line_length: int = 50,
                       max_line_gap: int = 20) -> Optional[np.ndarray]:
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
                           angle_tolerance: float = 5.0,
                           min_length: float = 100.0) -> List[np.ndarray]:
    """
    Filter to keep only approximately horizontal lines above minimum length.
    Useful for Müller-Lyer illusion where main shafts are horizontal.
    
    Args:
        lines: Array of detected lines
        angle_tolerance: Maximum deviation from horizontal (in degrees)
        min_length: Minimum line length to keep (pixels)
        
    Returns:
        List of horizontal lines
    """
    if lines is None:
        return []
    
    horizontal_lines = []
    
    for line in lines:
        coords = line[0]
        angle = calculate_line_angle(coords)
        length = calculate_line_length(coords)
        
        # Check if close to 0° or 180° (horizontal) AND above minimum length
        if (angle < angle_tolerance or angle > (180 - angle_tolerance)) and length >= min_length:
            horizontal_lines.append(coords)
    
    return horizontal_lines


def merge_similar_lines(lines: List[np.ndarray],
                       distance_threshold: float = 100.0,
                       angle_threshold: float = 5.0) -> List[np.ndarray]:
    """
    Merge lines that are collinear and overlapping.
    Helps reduce duplicate detections by combining fragments of the same line.
    
    Args:
        lines: List of line coordinates
        distance_threshold: Maximum distance between line midpoints to consider merging
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
            
            # Check if similar angle (collinear)
            angle_diff = abs(prop1['angle'] - prop2['angle'])
            
            # For horizontal lines, also check y-coordinates are close
            y_diff = abs(prop1['midpoint'][1] - prop2['midpoint'][1])
            
            # Check if lines are on same horizontal line and have similar angle
            if angle_diff < angle_threshold and y_diff < 20:  # 20px tolerance for y-coordinate
                # Additional check: see if projections overlap
                if lines_overlap_horizontally(line1, line2):
                    group.append(line2)
                    used.add(j)
        
        # Merge group into single line by combining endpoints
        if len(group) > 1:
            merged_line = merge_horizontal_group(group)
            merged.append(merged_line)
        else:
            merged.append(group[0])
    
    return merged


def lines_overlap_horizontally(line1: np.ndarray, line2: np.ndarray) -> bool:
    """
    Check if two horizontal line segments overlap or are close.
    
    Args:
        line1, line2: Line coordinates [x1, y1, x2, y2]
        
    Returns:
        True if lines overlap or are within 50px of each other
    """
    x1_min, x1_max = sorted([line1[0], line1[2]])
    x2_min, x2_max = sorted([line2[0], line2[2]])
    
    # Check if intervals overlap or are close
    gap = min(x1_max, x2_max) - max(x1_min, x2_min)
    return gap > -50  # Allow 50px gap between fragments


def merge_horizontal_group(group: List[np.ndarray]) -> np.ndarray:
    """
    Merge a group of horizontal line segments into one continuous line.
    
    Args:
        group: List of collinear horizontal line segments
        
    Returns:
        Merged line coordinates
    """
    # Find the min and max x coordinates across all segments
    all_x = []
    y_coords = []
    
    for line in group:
        all_x.extend([line[0], line[2]])
        y_coords.extend([line[1], line[3]])
    
    # Use the leftmost and rightmost points
    min_x = min(all_x)
    max_x = max(all_x)
    
    # Use average y-coordinate for the merged line
    avg_y = sum(y_coords) / len(y_coords)
    
    return np.array([min_x, avg_y, max_x, avg_y])


def assign_spatial_line_ids(lines: List[np.ndarray]) -> List[Dict]:
    """
    Assign line IDs based on spatial position (top-to-bottom, then left-to-right).
    This ensures consistent line numbering regardless of detection order.
    
    Args:
        lines: List of line coordinates
        
    Returns:
        List of line dictionaries with spatial IDs
    """
    if not lines:
        return []
    
    # Get properties for all lines
    line_props = []
    for line in lines:
        props = get_line_properties(line)
        line_props.append(props)
    
    # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
    # Use midpoint for sorting
    sorted_lines = sorted(line_props, key=lambda x: (x['midpoint'][1], x['midpoint'][0]))
    
    # Assign IDs based on spatial order
    results = []
    for i, props in enumerate(sorted_lines):
        props['id'] = f"line_{i}"
        results.append(props)
    
    return results


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
    
    # Assign spatial IDs and get properties
    results = assign_spatial_line_ids(lines)
    
    return results