#!/usr/bin/env python3
"""
Visualize shape comparison results with annotations.
Saves debug images showing detected shapes, classifications, and size comparisons.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from geometry_tools import (
    compare_shapes_in_image,
    load_image
)

def draw_shape_with_info(image, shapes, comparison_result, expected_larger, title):
    """
    Draw shapes with annotations on the image.
    
    Args:
        image: Input image (RGB)
        shapes: List of shape dictionaries
        comparison_result: Comparison result dictionary
        expected_larger: Expected result from filename
        title: Title for the image
    
    Returns:
        Annotated image (RGB)
    """
    # Convert RGB to BGR for OpenCV drawing
    vis_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    
    # Draw shapes with different colors
    colors = [(0, 255, 0), (0, 0, 255)]  # Green for shape 1, Red for shape 2
    
    for i, shape in enumerate(shapes[:2]):  # Only draw first 2 shapes
        contour = shape['contour']
        color = colors[i % len(colors)]
        
        # Draw contour
        cv2.drawContours(vis_image, [contour], -1, color, 2)
        
        # Draw centroid
        centroid = shape['centroid']
        cv2.circle(vis_image, (int(centroid[0]), int(centroid[1])), 5, color, -1)
        
        # Add shape label
        label = f"Shape {i+1}: {shape['type']}\nArea: {shape['area']:.0f}"
        label_position = (int(centroid[0]) + 10, int(centroid[1]) - 10)
        
        # Draw text background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Split text into lines
        lines = label.split('\n')
        for j, line in enumerate(lines):
            y_offset = j * 20
            cv2.putText(vis_image, line, 
                       (label_position[0], label_position[1] + y_offset),
                       font, font_scale, color, thickness)
    
    # Add comparison result at top
    shape_1_larger = comparison_result['is_shape_1_larger']
    area_ratio = comparison_result['area_ratio']
    
    result_text = f"Shape 1 larger: {shape_1_larger} (ratio: {area_ratio:.3f})"
    expected_text = f"Expected: {expected_larger}"
    status = "✅ CORRECT" if shape_1_larger == expected_larger else "❌ INCORRECT"
    
    # Draw title and results
    cv2.putText(vis_image, title, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image, result_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_image, expected_text, (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_image, status, (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if shape_1_larger == expected_larger else (0, 0, 255), 2)
    
    # Convert back to RGB
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

def visualize_shape_comparison():
    """Create visual debug images for shape comparison."""
    
    print("=== Creating Shape Comparison Debug Images ===\n")
    
    # Test cases from different categories
    test_cases = [
        ("circle_100/circle2_YES_000.png", True, "Circle Comparison - YES"),
        ("circle_100/circle2_NO_025.png", False, "Circle Comparison - NO"),
        ("square_50/square_YES_000.png", True, "Square Comparison - YES"),
        ("square_50/square_NO_025.png", False, "Square Comparison - NO"),
        ("rectangle_50/rectangle_YES_000.png", True, "Rectangle Comparison - YES"),
        ("rectangle_50/rectangle_NO_025.png", False, "Rectangle Comparison - NO"),
    ]
    
    base_path = "/Users/yingxizhao/Desktop/visualIllusion/2821_finalProj_IMAGE_DATA/vision_2d_Compare_Size"
    debug_dir = "/Users/yingxizhao/Desktop/visualIllusion/debug_images"
    
    # Ensure debug directory exists
    os.makedirs(debug_dir, exist_ok=True)
    
    for filename, expected_larger, title in test_cases:
        print(f"Processing {filename}...")
        
        try:
            image_path = os.path.join(base_path, filename)
            
            # Load image
            image = load_image(image_path)
            
            # Get shape comparison result
            result = compare_shapes_in_image(image_path)
            
            if not result['success']:
                print(f"  ❌ Failed: {result['error']}")
                continue
            
            # Create visualization
            vis_image = draw_shape_with_info(
                image, 
                result['shapes'], 
                result, 
                expected_larger,
                title
            )
            
            # Save debug image
            output_filename = f"debug_{filename.replace('/', '_')}"
            output_path = os.path.join(debug_dir, output_filename)
            
            # Convert RGB back to BGR for OpenCV saving
            save_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, save_image)
            
            print(f"  ✅ Saved: {output_path}")
            print(f"     Shapes: {result['num_shapes']} detected")
            print(f"     Shape 1: {result['shape_1']['type']} (area: {result['shape_1']['area']:.0f})")
            print(f"     Shape 2: {result['shape_2']['type']} (area: {result['shape_2']['area']:.0f})")
            print(f"     Result: Shape 1 larger = {result['is_shape_1_larger']}, Expected = {expected_larger}")
            print(f"     Status: {'✅ CORRECT' if result['is_shape_1_larger'] == expected_larger else '❌ INCORRECT'}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
    
    print("All debug images saved to:", debug_dir)
    print("You can view the images to see shape detection results with annotations.")

if __name__ == "__main__":
    visualize_shape_comparison()
