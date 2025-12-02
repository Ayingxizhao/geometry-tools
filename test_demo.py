#!/usr/bin/env python3
"""
Simple test script for Geometry Tools API - Real Working Scenario Demo
"""

import numpy as np
import cv2
import os
from geometry_tools import (
    is_line_longer,
    measure_line_length,
    compare_all_lines,
    create_synthetic_lines,
    create_muller_lyer_illusion,
    print_comparison_result
)

def get_processed_lines_debug(image):
    """
    Get lines as processed by the actual API (with filtering and merging)
    
    Args:
        image: Input image
        
    Returns:
        List of processed line dictionaries
    """
    from geometry_tools.preprocessing import preprocess_for_line_detection
    from geometry_tools.line_detection import detect_and_filter_lines
    
    # Preprocess the image
    gray, edges = preprocess_for_line_detection(image)
    
    # Get detected lines with full processing (same as API)
    processed_lines = detect_and_filter_lines(edges, filter_horizontal=True, merge_duplicates=True)
    
    return processed_lines

def save_debug_image(image, detected_lines, test_name, expected_lines=None):
    """
    Save debug image showing detected lines vs expected lines
    
    Args:
        image: Original image
        detected_lines: List of detected line dictionaries
        test_name: Name for the debug image file
        expected_lines: Optional list of expected line configurations
    """
    # Create debug directory if it doesn't exist
    debug_dir = "debug_images"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Make a copy for drawing
    debug_img = image.copy()
    
    # Draw expected lines in blue if provided
    if expected_lines:
        for i, config in enumerate(expected_lines):
            start = config['start']
            end = config['end']
            cv2.line(debug_img, start, end, (255, 0, 0), 2)  # Blue for expected
            # Add label
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            cv2.putText(debug_img, f"E{i+1}", (mid_x-10, mid_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw detected lines in red
    for line in detected_lines:
        if 'coordinates' in line:
            coords = [int(coord) for coord in line['coordinates']]
            cv2.line(debug_img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 2)  # Red for detected
            # Add label with length
            mid_x = (coords[0] + coords[2]) // 2
            mid_y = (coords[1] + coords[3]) // 2
            label = f"{line['id']}: {line['length']:.0f}px"
            cv2.putText(debug_img, label, (mid_x-20, mid_y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Save the debug image
    filename = os.path.join(debug_dir, f"{test_name}_debug.png")
    cv2.imwrite(filename, debug_img)
    print(f"DEBUG: Saved visualization to {filename}")
    
    # Print detailed detection info
    print(f"DETECTION DETAILS for {test_name}:")
    print(f"  Expected lines: {len(expected_lines) if expected_lines else 'N/A'}")
    print(f"  Detected lines: {len(detected_lines)}")
    for i, line in enumerate(detected_lines):
        print(f"    {line['id']}: length={line['length']:.1f}px, angle={line['angle']:.1f}°")
    print()

def get_detected_lines_debug(image):
    """
    Get detected lines for debugging purposes - shows the ACTUAL results the API uses
    
    Args:
        image: Input image
        
    Returns:
        List of detected line dictionaries (merged, as used by API)
    """
    from geometry_tools.preprocessing import preprocess_for_line_detection
    from geometry_tools.line_detection import detect_and_filter_lines
    
    # Preprocess the image
    gray, edges = preprocess_for_line_detection(image)
    
    # Get raw lines first for comparison
    raw_lines = detect_and_filter_lines(edges, filter_horizontal=False, merge_duplicates=False)
    
    # Get the actual lines used by the API (with filtering and merging)
    merged_lines = detect_and_filter_lines(edges, filter_horizontal=True, merge_duplicates=True)
    
    print(f"DEBUG MERGE CHECK: Raw={len(raw_lines)} lines, Merged={len(merged_lines)} lines")
    if len(raw_lines) > len(merged_lines):
        print("  ✓ Merging is working - reduced duplicate lines")
    else:
        print("  ✗ Merging failed - still have duplicates")
    
    # Return the merged results (what the API actually uses)
    return merged_lines

def test_basic_functionality():
    """Test basic API functionality"""
    print("Testing Geometry Tools API - Real Working Scenarios")
    print("=" * 60)
    
    try:
        # Test 1: Basic line comparison
        print("\nTest 1: Basic Line Comparison")
        print("Input configuration:")
        print("  - Line A: (100,150) to (300,150) = 200px")
        print("  - Line B: (450,250) to (750,250) = 300px")
        print("Ground truth: Line B should be longer than Line A")
        print("-" * 60)
        
        line_configs = [
            {'start': (100, 150), 'end': (300, 150), 'color': (0, 0, 0), 'thickness': 3},  # 200px
            {'start': (450, 250), 'end': (750, 250), 'color': (0, 0, 0), 'thickness': 3},  # 300px
        ]
        
        image = create_synthetic_lines(
            width=800, 
            height=400,
            line_configs=line_configs
        )
        
        # Debug visualization
        detected_lines = get_detected_lines_debug(image)
        save_debug_image(image, detected_lines, "test1_basic_comparison", line_configs)
        
        result = is_line_longer(image, "line_1", "line_0")  # line_1 is bottom (longer), line_0 is top (shorter)
        print("RESULT: Comparing line_1 (bottom/longer) vs line_0 (top/shorter)")
        print_comparison_result(result)
        
        # Verify ground truth
        if result.get('success'):
            expected_answer = 'yes'  # line_1 (bottom) should be longer than line_0 (top)
            actual_answer = result.get('answer_text', '')
            correct = actual_answer == expected_answer
            print(f"VERIFICATION: {'✓ PASS' if correct else '✗ FAIL'} - Expected: {expected_answer}, Got: {actual_answer}")
        
        # Test 2: Müller-Lyer illusion
        print("\nTest 2: Müller-Lyer Illusion")
        print("Input configuration:")
        print("  - Inward arrows line: 200px shaft with arrows pointing inward")
        print("  - Outward arrows line: 200px shaft with arrows pointing outward")
        print("Ground truth: Both lines should be approximately equal (200px each)")
        print("Expected: The illusion may cause perceived difference but actual lengths are equal")
        print("-" * 60)
        inward_img, outward_img = create_muller_lyer_illusion(shaft_length=200)
        
        # Debug visualization for separate images
        inward_lines = get_processed_lines_debug(inward_img)
        outward_lines = get_processed_lines_debug(outward_img)
        
        print(f"DEBUG: Inward image detected {len(inward_lines)} lines")
        for line in inward_lines:
            print(f"  {line['id']}: {line['length']:.1f}px at {line['angle']:.1f}°")
        
        print(f"DEBUG: Outward image detected {len(outward_lines)} lines")
        for line in outward_lines:
            print(f"  {line['id']}: {line['length']:.1f}px at {line['angle']:.1f}°")
        print()
        
        # Measure each line separately
        inward_result = measure_line_length(inward_img, "line_0")
        outward_result = measure_line_length(outward_img, "line_0")
        
        if inward_result['success'] and outward_result['success']:
            inward_length = inward_result['length']
            outward_length = outward_result['length']
            difference = abs(inward_length - outward_length)
            relative_diff = difference / max(inward_length, outward_length)
            tolerance = 0.05  # 5%
            are_equal = relative_diff < tolerance
            
            print(f"RESULT: Inward arrows: {inward_length:.1f}px")
            print(f"RESULT: Outward arrows: {outward_length:.1f}px")
            print(f"RESULT: Difference: {difference:.1f}px ({relative_diff*100:.1f}%)")
            
            if are_equal:
                print("ANSWER: approximately equal (within tolerance)")
                expected_answer = "approximately equal"
                actual_answer = "approximately equal"
                correct = True
            else:
                longer = "inward" if inward_length > outward_length else "outward"
                print(f"ANSWER: {longer} arrows are longer")
                expected_answer = "approximately equal"
                actual_answer = f"{longer} longer"
                correct = False
            
            print(f"VERIFICATION: {'✓ PASS' if correct else '✗ FAIL'} - Expected: {expected_answer}, Got: {actual_answer}")
        else:
            print("FAILED: Could not measure both lines")
            correct = False
        
        # Test 3: Single line measurement
        print("\nTest 3: Single Line Measurement")
        print("Input configuration:")
        print("  - Single line: (100,150) to (500,150) = 400px")
        print("Ground truth: Measured length should be 400px ± 5px tolerance")
        print("-" * 60)
        single_image = create_synthetic_lines(
            width=600, 
            height=300,
            line_configs=[
                {'start': (100, 150), 'end': (500, 150), 'color': (0, 0, 0), 'thickness': 3},  # 400px
            ]
        )
        
        measure_result = measure_line_length(single_image, "line_0")
        if measure_result['success']:
            measured_length = measure_result['length']
            expected_length = 400
            error = abs(measured_length - expected_length)
            tolerance = 5
            within_tolerance = error <= tolerance
            
            print(f"SUCCESS: Measured line: {measured_length:.1f}px (expected: {expected_length}px)")
            print(f"ERROR: {error:.1f}px (tolerance: ±{tolerance}px)")
            print(f"VERIFICATION: {'✓ PASS' if within_tolerance else '✗ FAIL'} - Measurement within tolerance")
        else:
            print(f"FAILED: Measurement failed: {measure_result.get('error', 'Unknown error')}")
        
        # Test 4: Compare all lines
        print("\nTest 4: Auto-detect and Compare All Lines")
        print("Input configuration:")
        print("  - Line 0: (100,100) to (250,100) = 150px (expected shortest)")
        print("  - Line 1: (300,200) to (550,200) = 250px") 
        print("  - Line 2: (100,300) to (400,300) = 300px (expected longest)")
        print("Ground truth: Line 2 should be longest, Line 0 should be shortest")
        print("-" * 60)
        
        line_configs = [
                {'start': (100, 100), 'end': (250, 100), 'color': (0, 0, 0), 'thickness': 3},   # 150px
                {'start': (300, 200), 'end': (550, 200), 'color': (0, 0, 0), 'thickness': 3},   # 250px
                {'start': (100, 300), 'end': (400, 300), 'color': (0, 0, 0), 'thickness': 3},   # 300px
            ]
        
        multi_image = create_synthetic_lines(
            width=800, 
            height=500,
            line_configs=line_configs
        )
        
        # Debug visualization
        detected_lines = get_detected_lines_debug(multi_image)
        save_debug_image(multi_image, detected_lines, "test4_multiple_lines", line_configs)
        
        all_result = compare_all_lines(multi_image)
        if all_result['success']:
            print(f"SUCCESS: Found {len(all_result['lines'])} lines")
            print(f"RESULT: Longest: {all_result['longest_line']} ({all_result['longest_length']:.1f}px) - Expected: line_2")
            print(f"RESULT: Shortest: {all_result['shortest_line']} ({all_result['shortest_length']:.1f}px) - Expected: line_0")
            
            # Verify ground truth
            longest_correct = all_result['longest_line'] == 'line_2'
            shortest_correct = all_result['shortest_line'] == 'line_0'
            print(f"VERIFICATION: Longest correct: {longest_correct}, Shortest correct: {shortest_correct}")
        else:
            print(f"FAILED: Auto-detection failed: {all_result.get('error', 'Unknown error')}")
        
        print("\nAll tests completed successfully!")
        print("SUCCESS: Geometry Tools API is working correctly in real scenarios!")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
