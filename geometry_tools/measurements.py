"""
High-level measurement and comparison APIs.
These are the main functions that LLM-generated code will call.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from .preprocessing import load_image, preprocess_for_line_detection
from .line_detection import detect_and_filter_lines
from .triangle_detection import detect_and_classify_triangles as detect_triangles_internal, validate_triangle


def measure_line_length(image: Union[str, np.ndarray], 
                       line_id: str,
                       filter_horizontal: bool = True) -> Dict:
    """
    Measure the length of a specific line in an image.
    
    Args:
        image: Image path (str) or numpy array
        line_id: ID of the line to measure (e.g., "line_0", "line_1")
        filter_horizontal: Whether to filter for horizontal lines only
        
    Returns:
        Dictionary with:
            - success: bool
            - length: float (pixels) if found
            - line_id: str
            - properties: dict with full line properties
            - error: str if failed
    """
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Detect lines
    gray, edges = preprocess_for_line_detection(image)
    lines = detect_and_filter_lines(edges, filter_horizontal=filter_horizontal)
    
    # Find the requested line
    for line_props in lines:
        if line_props['id'] == line_id:
            return {
                'success': True,
                'length': line_props['length'],
                'line_id': line_id,
                'properties': line_props
            }
    
    # Line not found
    return {
        'success': False,
        'line_id': line_id,
        'error': f"Line '{line_id}' not found. Available: {[l['id'] for l in lines]}"
    }


def is_line_longer(image: Union[str, np.ndarray],
                   line_A_id: str,
                   line_B_id: str,
                   tolerance: float = 0.05,
                   filter_horizontal: bool = True) -> Dict:
    """
    Compare two lines: Is line A longer than line B?
    Returns yes/no answer suitable for LLM responses.
    
    Args:
        image: Image path (str) or numpy array
        line_A_id: ID of first line (e.g., "line_0")
        line_B_id: ID of second line (e.g., "line_1")
        tolerance: Relative tolerance for considering lines equal (default 5%)
        filter_horizontal: Whether to filter for horizontal lines only
        
    Returns:
        Dictionary with:
            - answer: bool (True if A > B, False if B >= A)
            - answer_text: str ("yes" or "no")
            - are_equal: bool (True if within tolerance)
            - line_A_length: float
            - line_B_length: float
            - difference: float (absolute difference in pixels)
            - relative_difference: float (percentage)
            - confidence: str ("high", "medium", "low")
    """
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Detect lines
    gray, edges = preprocess_for_line_detection(image)
    lines = detect_and_filter_lines(edges, filter_horizontal=filter_horizontal)
    
    # Find both lines
    line_A = None
    line_B = None
    
    for line_props in lines:
        if line_props['id'] == line_A_id:
            line_A = line_props
        if line_props['id'] == line_B_id:
            line_B = line_props
    
    # Check if both found
    if line_A is None or line_B is None:
        available = [l['id'] for l in lines]
        missing = []
        if line_A is None:
            missing.append(line_A_id)
        if line_B is None:
            missing.append(line_B_id)
        
        return {
            'success': False,
            'error': f"Lines not found: {missing}. Available: {available}"
        }
    
    # Get lengths
    length_A = line_A['length']
    length_B = line_B['length']
    
    # Calculate differences
    abs_diff = abs(length_A - length_B)
    max_length = max(length_A, length_B)
    rel_diff = abs_diff / max_length if max_length > 0 else 0
    
    # Check if equal within tolerance
    are_equal = rel_diff < tolerance
    
    # Determine answer
    if are_equal:
        answer = False  # Not definitively longer
        answer_text = "approximately equal"
        confidence = "low"
    else:
        answer = length_A > length_B
        answer_text = "yes" if answer else "no"
        
        # Confidence based on relative difference
        if rel_diff > 0.15:  # >15% difference
            confidence = "high"
        elif rel_diff > 0.08:  # 8-15% difference
            confidence = "medium"
        else:  # 5-8% difference
            confidence = "low"
    
    return {
        'success': True,
        'answer': answer,
        'answer_text': answer_text,
        'are_equal': are_equal,
        'line_A_length': length_A,
        'line_B_length': length_B,
        'difference': abs_diff,
        'relative_difference': rel_diff,
        'confidence': confidence,
        'comparison': f"{line_A_id} ({'longer' if length_A > length_B else 'shorter'}) than {line_B_id}"
    }


def compare_all_lines(image: Union[str, np.ndarray],
                     filter_horizontal: bool = True,
                     tolerance: float = 0.05) -> Dict:
    """
    Detect all lines and compare their lengths.
    Useful when line IDs are not known in advance.
    
    Args:
        image: Image path (str) or numpy array
        filter_horizontal: Whether to filter for horizontal lines only
        tolerance: Relative tolerance for considering lines equal
        
    Returns:
        Dictionary with:
            - num_lines: int
            - lines: list of line properties sorted by length
            - longest_line: str (line_id)
            - shortest_line: str (line_id)
            - pairwise_comparisons: list of comparison results
    """
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Detect lines
    gray, edges = preprocess_for_line_detection(image)
    lines = detect_and_filter_lines(edges, filter_horizontal=filter_horizontal)
    
    if not lines:
        return {
            'success': False,
            'error': 'No lines detected in image'
        }
    
    # Sort by length
    sorted_lines = sorted(lines, key=lambda x: x['length'], reverse=True)
    
    # Get longest and shortest
    longest = sorted_lines[0]
    shortest = sorted_lines[-1]
    
    # Perform pairwise comparisons
    comparisons = []
    for i in range(len(sorted_lines)):
        for j in range(i + 1, len(sorted_lines)):
            line_a = sorted_lines[i]
            line_b = sorted_lines[j]
            
            length_diff = line_a['length'] - line_b['length']
            rel_diff = length_diff / line_a['length']
            
            comparisons.append({
                'line_A': line_a['id'],
                'line_B': line_b['id'],
                'length_A': line_a['length'],
                'length_B': line_b['length'],
                'difference': length_diff,
                'relative_difference': rel_diff,
                'A_longer_than_B': rel_diff > tolerance
            })
    
    return {
        'success': True,
        'num_lines': len(lines),
        'lines': sorted_lines,
        'longest_line': longest['id'],
        'shortest_line': shortest['id'],
        'longest_length': longest['length'],
        'shortest_length': shortest['length'],
        'pairwise_comparisons': comparisons
    }


def answer_comparison_question(image: Union[str, np.ndarray],
                               question: str,
                               line_A_id: Optional[str] = None,
                               line_B_id: Optional[str] = None,
                               tolerance: float = 0.05) -> Dict:
    """
    Answer a natural language question about line comparison.
    Helper function for more flexible querying.
    
    Args:
        image: Image path or numpy array
        question: Question string (e.g., "Is line A longer than line B?")
        line_A_id: Optional explicit line A ID
        line_B_id: Optional explicit line B ID
        tolerance: Relative tolerance for equality
        
    Returns:
        Dictionary with answer and supporting data
    """
    question_lower = question.lower()
    
    # If line IDs not provided, try to auto-detect
    if line_A_id is None or line_B_id is None:
        # Simple heuristic: look for "line_0", "line_1" etc in question
        import re
        found_ids = re.findall(r'line[_\s](\d+)', question_lower)
        
        if len(found_ids) >= 2:
            line_A_id = f"line_{found_ids[0]}"
            line_B_id = f"line_{found_ids[1]}"
        else:
            # Auto-detect all lines
            all_lines = compare_all_lines(image, tolerance=tolerance)
            if not all_lines['success']:
                return all_lines
            
            # Use first two lines
            if all_lines['num_lines'] >= 2:
                line_A_id = all_lines['lines'][0]['id']
                line_B_id = all_lines['lines'][1]['id']
            else:
                return {
                    'success': False,
                    'error': 'Could not identify lines to compare'
                }
    
    # Perform comparison
    result = is_line_longer(image, line_A_id, line_B_id, tolerance=tolerance)
    
    # Add natural language response
    if result.get('success'):
        if result['are_equal']:
            result['natural_answer'] = f"The lines are approximately equal in length (within {tolerance*100}% tolerance)."
        else:
            longer_line = line_A_id if result['answer'] else line_B_id
            result['natural_answer'] = f"Yes, {longer_line} is longer." if result['answer'] else f"No, {line_B_id} is longer."
    
    return result


def detect_and_classify_triangles(image: Union[str, np.ndarray],
                                 min_area: float = 1000.0,
                                 max_area: float = 100000.0) -> Dict:
    """
    Detect all triangles in an image and classify their types.
    
    Args:
        image: Image path (str) or numpy array
        min_area: Minimum triangle area to consider (pixels)
        max_area: Maximum triangle area to consider (pixels)
        
    Returns:
        Dictionary with:
            - success: bool
            - num_triangles: int
            - triangles: list of triangle properties with classifications
            - equilateral_triangles: list of equilateral triangle IDs
            - right_triangles: list of right triangle IDs
            - error: str if failed
    """
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Preprocess for edge detection
    gray, edges = preprocess_for_line_detection(image)
    
    # Detect and classify triangles
    triangles = detect_triangles_internal(edges, min_area, max_area)
    
    if not triangles:
        return {
            'success': False,
            'error': 'No triangles detected in image'
        }
    
    # Extract triangle types
    equilateral_ids = []
    right_ids = []
    
    for triangle in triangles:
        triangle_type = triangle['classification']['type']
        if triangle_type == 'equilateral':
            equilateral_ids.append(triangle['id'])
        elif triangle_type == 'right':
            right_ids.append(triangle['id'])
    
    return {
        'success': True,
        'num_triangles': len(triangles),
        'triangles': triangles,
        'equilateral_triangles': equilateral_ids,
        'right_triangles': right_ids
    }


def is_valid_equilateral(image: Union[str, np.ndarray],
                         triangle_id: str,
                         angle_tolerance: float = 5.0,
                         side_ratio_tolerance: float = 0.05) -> Dict:
    """
    Check if a specific triangle is a valid equilateral triangle.
    
    Args:
        image: Image path (str) or numpy array
        triangle_id: ID of the triangle to validate (e.g., "triangle_0")
        angle_tolerance: Tolerance for angle matching in degrees (±5% default)
        side_ratio_tolerance: Tolerance for side ratio equality (5% default)
        
    Returns:
        Dictionary with:
            - success: bool
            - is_valid: bool (True if valid equilateral)
            - triangle_id: str
            - confidence: str ("high", "medium", "low")
            - classification: dict with full triangle properties
            - error: str if failed
    """
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Detect triangles
    gray, edges = preprocess_for_line_detection(image)
    triangles = detect_triangles_internal(edges)
    
    # Find the requested triangle
    target_triangle = None
    for triangle in triangles:
        if triangle['id'] == triangle_id:
            target_triangle = triangle
            break
    
    if target_triangle is None:
        available = [t['id'] for t in triangles]
        return {
            'success': False,
            'triangle_id': triangle_id,
            'error': f"Triangle '{triangle_id}' not found. Available: {available}"
        }
    
    # Validate as equilateral
    vertices = np.array(target_triangle['vertices'])
    validation = validate_triangle(
        vertices, 
        "equilateral", 
        angle_tolerance, 
        side_ratio_tolerance
    )
    
    return {
        'success': True,
        'is_valid': validation['is_valid'],
        'triangle_id': triangle_id,
        'confidence': validation['confidence'],
        'classification': target_triangle['classification']
    }


def is_valid_right_triangle(image: Union[str, np.ndarray],
                           triangle_id: str,
                           angle_tolerance: float = 5.0) -> Dict:
    """
    Check if a specific triangle is a valid right triangle.
    
    Args:
        image: Image path (str) or numpy array
        triangle_id: ID of the triangle to validate (e.g., "triangle_0")
        angle_tolerance: Tolerance for angle matching in degrees (±5% default)
        
    Returns:
        Dictionary with:
            - success: bool
            - is_valid: bool (True if valid right triangle)
            - triangle_id: str
            - confidence: str ("high", "medium", "low")
            - classification: dict with full triangle properties
            - error: str if failed
    """
    # Load image if path provided
    if isinstance(image, str):
        image = load_image(image)
    
    # Detect triangles
    gray, edges = preprocess_for_line_detection(image)
    triangles = detect_triangles_internal(edges)
    
    # Find the requested triangle
    target_triangle = None
    for triangle in triangles:
        if triangle['id'] == triangle_id:
            target_triangle = triangle
            break
    
    if target_triangle is None:
        available = [t['id'] for t in triangles]
        return {
            'success': False,
            'triangle_id': triangle_id,
            'error': f"Triangle '{triangle_id}' not found. Available: {available}"
        }
    
    # Validate as right triangle
    vertices = np.array(target_triangle['vertices'])
    validation = validate_triangle(vertices, "right", angle_tolerance)
    
    return {
        'success': True,
        'is_valid': validation['is_valid'],
        'triangle_id': triangle_id,
        'confidence': validation['confidence'],
        'classification': target_triangle['classification']
    }


def answer_triangle_question(image: Union[str, np.ndarray],
                             question: str,
                             triangle_id: Optional[str] = None,
                             angle_tolerance: float = 5.0,
                             side_ratio_tolerance: float = 0.05) -> Dict:
    """
    Answer a natural language question about triangle validation.
    Helper function for more flexible querying.
    
    Args:
        image: Image path or numpy array
        question: Question string (e.g., "Is this a valid equilateral triangle?")
        triangle_id: Optional explicit triangle ID
        angle_tolerance: Tolerance for angle matching in degrees
        side_ratio_tolerance: Tolerance for side ratio equality
        
    Returns:
        Dictionary with answer and supporting data
    """
    question_lower = question.lower()
    
    # Determine triangle type from question
    if "equilateral" in question_lower:
        expected_type = "equilateral"
    elif "right" in question_lower:
        expected_type = "right"
    else:
        return {
            'success': False,
            'error': 'Could not determine triangle type from question. Please specify "equilateral" or "right".'
        }
    
    # If triangle ID not provided, try to auto-detect
    if triangle_id is None:
        # Auto-detect all triangles
        all_triangles = detect_and_classify_triangles(image)
        if not all_triangles['success']:
            return all_triangles
        
        # Use first triangle
        if all_triangles['num_triangles'] >= 1:
            triangle_id = all_triangles['triangles'][0]['id']
        else:
            return {
                'success': False,
                'error': 'No triangles found in image'
            }
    
    # Perform validation based on type
    if expected_type == "equilateral":
        result = is_valid_equilateral(image, triangle_id, angle_tolerance, side_ratio_tolerance)
    else:  # right triangle
        result = is_valid_right_triangle(image, triangle_id, angle_tolerance)
    
    # Add natural language response
    if result.get('success'):
        if result['is_valid']:
            result['natural_answer'] = f"Yes, this is a valid {expected_type} triangle."
        else:
            result['natural_answer'] = f"No, this is not a valid {expected_type} triangle."
    
    return result