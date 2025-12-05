"""
Triangle detection utilities using contour analysis.
Detects triangular shapes and validates their geometric properties.
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional


def detect_triangles_contours(edges: np.ndarray,
                             min_area: float = 1000.0,
                             max_area: float = 100000.0) -> List[np.ndarray]:
    """
    Detect triangles using contour detection and polygon approximation.
    
    Args:
        edges: Edge-detected image (from Canny)
        min_area: Minimum contour area to consider (pixels)
        max_area: Maximum contour area to consider (pixels)
        
    Returns:
        List of detected triangles, each as array of 3 vertices [[x1,y1], [x2,y2], [x3,y3]]
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    triangles = []
    
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's a triangle (3 vertices)
        if len(approx) == 3:
            # Extract vertices and convert to proper format
            vertices = approx.reshape(-1, 2)
            triangles.append(vertices)
    
    return triangles


def calculate_triangle_angles(vertices: np.ndarray) -> List[float]:
    """
    Calculate internal angles of a triangle from its vertices.
    
    Args:
        vertices: Array of 3 vertices [[x1,y1], [x2,y2], [x3,y3]]
        
    Returns:
        List of 3 internal angles in degrees
    """
    if len(vertices) != 3:
        raise ValueError("Triangle must have exactly 3 vertices")
    
    angles = []
    
    for i in range(3):
        # Get current vertex and adjacent vertices
        curr = vertices[i]
        prev = vertices[(i - 1) % 3]
        next_vert = vertices[(i + 1) % 3]
        
        # Calculate vectors from current vertex
        v1 = prev - curr
        v2 = next_vert - curr
        
        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 == 0 or mag2 == 0:
            angle = 0.0
        else:
            cos_angle = dot_product / (mag1 * mag2)
            # Clamp to avoid numerical errors
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.degrees(math.acos(cos_angle))
        
        angles.append(angle)
    
    return angles


def calculate_side_lengths(vertices: np.ndarray) -> List[float]:
    """
    Calculate side lengths of a triangle from its vertices.
    
    Args:
        vertices: Array of 3 vertices [[x1,y1], [x2,y2], [x3,y3]]
        
    Returns:
        List of 3 side lengths in pixels
    """
    if len(vertices) != 3:
        raise ValueError("Triangle must have exactly 3 vertices")
    
    sides = []
    for i in range(3):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 3]
        length = np.linalg.norm(p2 - p1)
        sides.append(length)
    
    return sides


def classify_triangle_type(vertices: np.ndarray, 
                          angle_tolerance: float = 5.0,
                          side_ratio_tolerance: float = 0.05) -> Dict:
    """
    Classify triangle type based on angles and side lengths.
    
    Args:
        vertices: Array of 3 vertices [[x1,y1], [x2,y2], [x3,y3]]
        angle_tolerance: Tolerance for angle matching in degrees
        side_ratio_tolerance: Tolerance for side ratio equality (default 5%)
        
    Returns:
        Dictionary with triangle classification and properties
    """
    # Calculate angles and sides
    angles = calculate_triangle_angles(vertices)
    sides = calculate_side_lengths(vertices)
    
    # Sort angles for easier classification
    sorted_angles = sorted(angles)
    
    # Check for right triangle (one angle ~90°)
    is_right = any(abs(angle - 90.0) <= angle_tolerance for angle in angles)
    
    # Check for equilateral triangle (all angles ~60° and all sides equal)
    angle_diff = max(angles) - min(angles)
    side_ratio = max(sides) / min(sides) if min(sides) > 0 else float('inf')
    
    is_equilateral = (angle_diff <= angle_tolerance and 
                     side_ratio <= (1.0 + side_ratio_tolerance))
    
    # Determine primary classification
    if is_equilateral:
        triangle_type = "equilateral"
    elif is_right:
        triangle_type = "right"
    else:
        triangle_type = "other"
    
    return {
        'type': triangle_type,
        'angles': angles,
        'sorted_angles': sorted_angles,
        'sides': sides,
        'is_equilateral': is_equilateral,
        'is_right': is_right,
        'angle_diff': angle_diff,
        'side_ratio': side_ratio,
        'vertices': vertices.tolist()
    }


def validate_triangle(vertices: np.ndarray,
                     expected_type: str,
                     angle_tolerance: float = 5.0,
                     side_ratio_tolerance: float = 0.05) -> Dict:
    """
    Validate if a triangle meets the criteria for a specific type.
    
    Args:
        vertices: Array of 3 vertices [[x1,y1], [x2,y2], [x3,y3]]
        expected_type: Type to validate ("equilateral" or "right")
        angle_tolerance: Tolerance for angle matching in degrees
        side_ratio_tolerance: Tolerance for side ratio equality (default 5%)
        
    Returns:
        Dictionary with validation result
    """
    classification = classify_triangle_type(vertices, angle_tolerance, side_ratio_tolerance)
    
    if expected_type == "equilateral":
        is_valid = classification['is_equilateral']
        confidence = calculate_equilateral_confidence(classification, angle_tolerance, side_ratio_tolerance)
    elif expected_type == "right":
        is_valid = classification['is_right']
        confidence = calculate_right_triangle_confidence(classification, angle_tolerance)
    else:
        is_valid = False
        confidence = "low"
    
    return {
        'is_valid': is_valid,
        'expected_type': expected_type,
        'detected_type': classification['type'],
        'confidence': confidence,
        'classification': classification
    }


def calculate_equilateral_confidence(classification: Dict, 
                                   angle_tolerance: float,
                                   side_ratio_tolerance: float) -> str:
    """
    Calculate confidence level for equilateral triangle validation.
    
    Args:
        classification: Triangle classification dictionary
        angle_tolerance: Angle tolerance in degrees
        side_ratio_tolerance: Side ratio tolerance
        
    Returns:
        Confidence level: "high", "medium", or "low"
    """
    angle_diff = classification['angle_diff']
    side_ratio = classification['side_ratio']
    
    # High confidence: very close to perfect equilateral
    if angle_diff <= angle_tolerance * 0.5 and side_ratio <= 1.02:
        return "high"
    # Medium confidence: within tolerance but not perfect
    elif angle_diff <= angle_tolerance * 0.8 and side_ratio <= (1.0 + side_ratio_tolerance * 0.8):
        return "medium"
    # Low confidence: barely within tolerance
    else:
        return "low"


def calculate_right_triangle_confidence(classification: Dict, 
                                      angle_tolerance: float) -> str:
    """
    Calculate confidence level for right triangle validation.
    
    Args:
        classification: Triangle classification dictionary
        angle_tolerance: Angle tolerance in degrees
        
    Returns:
        Confidence level: "high", "medium", or "low"
    """
    angles = classification['angles']
    
    # Find angle closest to 90°
    right_angle_error = min(abs(angle - 90.0) for angle in angles)
    
    # High confidence: very close to 90°
    if right_angle_error <= angle_tolerance * 0.3:
        return "high"
    # Medium confidence: reasonably close
    elif right_angle_error <= angle_tolerance * 0.7:
        return "medium"
    # Low confidence: barely within tolerance
    else:
        return "low"


def assign_triangle_ids(triangles: List[np.ndarray]) -> List[Dict]:
    """
    Assign triangle IDs based on spatial position (top-to-bottom, then left-to-right).
    
    Args:
        triangles: List of triangle vertex arrays
        
    Returns:
        List of triangle dictionaries with spatial IDs and properties
    """
    if not triangles:
        return []
    
    triangle_props = []
    
    for i, vertices in enumerate(triangles):
        # Calculate centroid
        centroid = np.mean(vertices, axis=0)
        
        # Get classification
        classification = classify_triangle_type(vertices)
        
        triangle_dict = {
            'id': f"triangle_{i}",
            'vertices': vertices.tolist(),
            'centroid': centroid.tolist(),
            'classification': classification
        }
        triangle_props.append(triangle_dict)
    
    # Sort by centroid position (top to bottom, then left to right)
    sorted_triangles = sorted(triangle_props, key=lambda x: (x['centroid'][1], x['centroid'][0]))
    
    # Reassign IDs based on spatial order
    for i, triangle in enumerate(sorted_triangles):
        triangle['id'] = f"triangle_{i}"
    
    return sorted_triangles


def detect_and_classify_triangles(edges: np.ndarray,
                                 min_area: float = 1000.0,
                                 max_area: float = 100000.0) -> List[Dict]:
    """
    Complete pipeline: detect triangles and return classified results.
    
    Args:
        edges: Edge-detected image
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider
        
    Returns:
        List of dictionaries, each containing triangle properties
    """
    # Detect triangles
    raw_triangles = detect_triangles_contours(edges, min_area, max_area)
    
    if not raw_triangles:
        return []
    
    # Assign spatial IDs and get properties
    results = assign_triangle_ids(raw_triangles)
    
    return results
