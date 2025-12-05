#!/usr/bin/env python3
"""
CORRECTED batch validation - properly checks ground truth:
- equilateral_YES: should be equilateral triangle
- equilateral_NO: should be triangle that is NOT equilateral (but still valid)
- right_YES: should be right triangle  
- right_NO: should be triangle that is NOT right (but still valid)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_tools import detect_and_classify_triangles, load_image

def correct_batch_validate():
    """Correctly validate against the actual ground truth meaning."""
    
    base_path = "/Users/yingxizhao/Desktop/visualIllusion/2821_finalProj_IMAGE_DATA/vision_2d_Check_Triangles"
    
    print("=== CORRECTED Ground Truth Validation ===\n")
    
    # Test equilateral files
    print("EQUILATERAL FILES:")
    print("-" * 50)
    
    equilateral_dir = os.path.join(base_path, "vision_equilateral_triangles_50")
    equilateral_correct = 0
    equilateral_total = 0
    equilateral_errors = []
    
    for filename in sorted(os.listdir(equilateral_dir)):
        if filename.endswith('.png'):
            expected_equilateral = "YES" in filename
            image_path = os.path.join(equilateral_dir, filename)
            
            try:
                result = detect_and_classify_triangles(image_path)
                
                if result['success']:
                    triangle = result['triangles'][0]
                    classification = triangle['classification']
                    actual_equilateral = classification['type'] == 'equilateral'
                    
                    # Ground truth: YES means should be equilateral, NO means should NOT be equilateral
                    correct = actual_equilateral == expected_equilateral
                    
                    if correct:
                        equilateral_correct += 1
                        status = "✅"
                    else:
                        equilateral_errors.append((filename, expected_equilateral, actual_equilateral, classification['type']))
                        status = "❌"
                    
                    equilateral_total += 1
                    
                    angles = classification['angles']
                    max_diff = max(abs(a - 60) for a in angles)
                    
                    print(f"  {status} {filename}: Is equilateral={actual_equilateral}, Expected={expected_equilateral}, Type={classification['type']}, Max diff={max_diff:.1f}°")
                    
                else:
                    equilateral_errors.append((filename, expected_equilateral, "NO_TRIANGLE", "No triangle detected"))
                    print(f"  ❌ {filename}: NO TRIANGLE DETECTED")
                    equilateral_total += 1
                    
            except Exception as e:
                equilateral_errors.append((filename, expected_equilateral, "ERROR", str(e)))
                print(f"  ❌ {filename}: ERROR - {e}")
                equilateral_total += 1
    
    equilateral_accuracy = (equilateral_correct / equilateral_total * 100) if equilateral_total > 0 else 0
    print(f"\nEquilateral Results: {equilateral_correct}/{equilateral_total} ({equilateral_accuracy:.1f}% accuracy)")
    
    if equilateral_errors:
        print("Equilateral Errors:")
        for filename, expected, actual, details in equilateral_errors:
            print(f"  {filename}: Expected equilateral={expected}, Got {actual}, Details={details}")
    
    print("\n" + "="*60 + "\n")
    
    # Test right triangle files
    print("RIGHT TRIANGLE FILES:")
    print("-" * 50)
    
    right_dir = os.path.join(base_path, "vision_right_triangles_50")
    right_correct = 0
    right_total = 0
    right_errors = []
    
    for filename in sorted(os.listdir(right_dir)):
        if filename.endswith('.png'):
            expected_right = "YES" in filename
            image_path = os.path.join(right_dir, filename)
            
            try:
                result = detect_and_classify_triangles(image_path)
                
                if result['success']:
                    triangle = result['triangles'][0]
                    classification = triangle['classification']
                    actual_right = classification['type'] == 'right'
                    
                    # Ground truth: YES means should be right triangle, NO means should NOT be right triangle
                    correct = actual_right == expected_right
                    
                    if correct:
                        right_correct += 1
                        status = "✅"
                    else:
                        right_errors.append((filename, expected_right, actual_right, classification['type']))
                        status = "❌"
                    
                    right_total += 1
                    
                    angles = classification['angles']
                    min_diff = min(abs(a - 90) for a in angles)
                    
                    print(f"  {status} {filename}: Is right={actual_right}, Expected={expected_right}, Type={classification['type']}, Min diff to 90°={min_diff:.1f}°")
                    
                else:
                    right_errors.append((filename, expected_right, "NO_TRIANGLE", "No triangle detected"))
                    print(f"  ❌ {filename}: NO TRIANGLE DETECTED")
                    right_total += 1
                    
            except Exception as e:
                right_errors.append((filename, expected_right, "ERROR", str(e)))
                print(f"  ❌ {filename}: ERROR - {e}")
                right_total += 1
    
    right_accuracy = (right_correct / right_total * 100) if right_total > 0 else 0
    print(f"\nRight Triangle Results: {right_correct}/{right_total} ({right_accuracy:.1f}% accuracy)")
    
    if right_errors:
        print("Right Triangle Errors:")
        for filename, expected, actual, details in right_errors:
            print(f"  {filename}: Expected right={expected}, Got {actual}, Details={details}")
    
    print("\n" + "="*60 + "\n")
    
    # Overall results
    total_correct = equilateral_correct + right_correct
    total_images = equilateral_total + right_total
    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    
    print("=== CORRECTED OVERALL RESULTS ===")
    print(f"Total Images: {total_images}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print(f"Equilateral Accuracy: {equilateral_accuracy:.1f}% ({equilateral_correct}/{equilateral_total})")
    print(f"Right Triangle Accuracy: {right_accuracy:.1f}% ({right_correct}/{right_total})")
    
    print(f"\n=== GROUND TRUTH MEANING ===")
    print("equilateral_YES: contains equilateral triangle")
    print("equilateral_NO: contains triangle that is NOT equilateral")
    print("right_YES: contains right triangle")
    print("right_NO: contains triangle that is NOT right")

if __name__ == "__main__":
    correct_batch_validate()
