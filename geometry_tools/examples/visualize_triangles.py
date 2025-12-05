#!/usr/bin/env python3
"""
Visualize triangle detection results with annotations.
Saves debug images showing detected triangles, angles, and classifications.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from geometry_tools import (
    detect_and_classify_triangles,
    is_valid_equilateral,
    is_valid_right_triangle,
    load_image
)

def draw_triangle_with_info(image, triangle, validation_result, expected_result, title):
    """Draw triangle with angle annotations and classification info."""
    
    # Make a copy for drawing
    vis_image = image.copy()
    if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
        # Convert BGR to RGB for matplotlib
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    elif len(vis_image.shape) == 2:
        # Convert grayscale to RGB
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    
    # Get triangle data
    vertices = np.array(triangle['vertices'], dtype=np.int32)
    angles = triangle['classification']['angles']
    triangle_type = triangle['classification']['type']
    
    # Draw triangle outline
    cv2.polylines(vis_image, [vertices], True, (0, 255, 0), 3)
    
    # Draw vertices
    for i, vertex in enumerate(vertices):
        cv2.circle(vis_image, tuple(vertex), 8, (255, 0, 0), -1)
        
        # Add angle labels near vertices
        angle_text = f"{angles[i]:.1f}°"
        cv2.putText(vis_image, angle_text, tuple(vertex - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add classification info
    height, width = vis_image.shape[:2]
    
    # Title
    cv2.putText(vis_image, title, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    # Triangle type
    type_text = f"Type: {triangle_type}"
    cv2.putText(vis_image, type_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Validation result
    if validation_result['success']:
        is_valid = validation_result['is_valid']
        confidence = validation_result['confidence']
        valid_text = f"Valid: {is_valid} (Confidence: {confidence})"
        color = (0, 255, 0) if is_valid == expected_result else (0, 0, 255)
    else:
        valid_text = f"Error: {validation_result['error']}"
        color = (0, 0, 255)
    
    cv2.putText(vis_image, valid_text, (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Expected vs Actual
    expected_text = f"Expected: {expected_result}"
    cv2.putText(vis_image, expected_text, (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add angle details
    angles_text = f"Angles: {[f'{a:.1f}°' for a in angles]}"
    cv2.putText(vis_image, angles_text, (10, height - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Side ratio for equilateral
    if triangle_type == 'equilateral' or 'equilateral' in title.lower():
        side_ratio = triangle['classification']['side_ratio']
        ratio_text = f"Side ratio: {side_ratio:.3f}"
        cv2.putText(vis_image, ratio_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return vis_image

def create_visualized_debug_images():
    """Create debug images with triangle detection visualization."""
    
    # Create debug directory
    debug_dir = "/Users/yingxizhao/Desktop/visualIllusion/debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Test cases
    test_cases = [
        ("equilateral_YES_000.png", "equilateral", True),
        ("equilateral_NO_025.png", "equilateral", False),
        ("equilateral_NO_028.png", "equilateral", False),  # Borderline case
        ("triangle2_YES_000.png", "right", True),
        ("triangle2_NO_025.png", "right", False),
        ("triangle2_NO_028.png", "right", False),  # Borderline case
    ]
    
    base_path = "/Users/yingxizhao/Desktop/visualIllusion/2821_finalProj_IMAGE_DATA/vision_2d_Check_Triangles"
    
    print("=== Creating Visualized Debug Images ===\n")
    
    for filename, triangle_type, expected_valid in test_cases:
        print(f"Processing {filename}...")
        
        # Determine directory
        if "equilateral" in filename:
            dir_path = os.path.join(base_path, "vision_equilateral_triangles_50")
        else:
            dir_path = os.path.join(base_path, "vision_right_triangles_50")
        
        image_path = os.path.join(dir_path, filename)
        
        try:
            # Load and process image
            image = load_image(image_path)
            
            # Detect triangles
            result = detect_and_classify_triangles(image)
            
            if not result['success']:
                print(f"  ❌ No triangles detected")
                continue
            
            # Get first triangle
            triangle = result['triangles'][0]
            
            # Validate based on type
            if triangle_type == "equilateral":
                validation = is_valid_equilateral(image, triangle['id'], angle_tolerance=5.0)
                title = f"Equilateral Triangle - {filename}"
            else:  # right triangle
                validation = is_valid_right_triangle(image, triangle['id'], angle_tolerance=5.0)
                title = f"Right Triangle - {filename}"
            
            # Create visualization
            vis_image = draw_triangle_with_info(image, triangle, validation, expected_valid, title)
            
            # Save debug image
            output_filename = f"debug_{filename}"
            output_path = os.path.join(debug_dir, output_filename)
            
            # Convert RGB back to BGR for OpenCV saving
            if len(vis_image.shape) == 3 and vis_image.shape[2] == 3:
                save_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            else:
                save_image = vis_image
            
            cv2.imwrite(output_path, save_image)
            print(f"  ✅ Saved: {output_path}")
            
            # Print summary
            angles = triangle['classification']['angles']
            if triangle_type == "equilateral":
                max_diff = max(abs(a - 60) for a in angles)
                print(f"     Angles: {[f'{a:.1f}°' for a in angles]} (max diff from 60°: {max_diff:.1f}°)")
            else:
                min_diff = min(abs(a - 90) for a in angles)
                print(f"     Angles: {[f'{a:.1f}°' for a in angles]} (min diff from 90°: {min_diff:.1f}°)")
            
            if validation['success']:
                is_valid = validation['is_valid']
                status = "✅ CORRECT" if is_valid == expected_valid else "❌ INCORRECT"
                print(f"     Result: {status} (Valid: {is_valid}, Expected: {expected_valid})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    print(f"All debug images saved to: {debug_dir}")
    print("You can view the images to see triangle detection results with angle annotations.")

if __name__ == "__main__":
    create_visualized_debug_images()
