#!/usr/bin/env python3
"""
Comprehensive batch validation for shape comparison on vision_2d_Compare_Size dataset.
Tests all available images to verify robustness and accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_tools import compare_shapes_in_image

def batch_validate_shape_comparison():
    """Validate shape comparison on all images in the Compare_Size dataset."""
    
    base_path = "/Users/yingxizhao/Desktop/visualIllusion/2821_finalProj_IMAGE_DATA/vision_2d_Compare_Size"
    
    print("=== COMPREHENSIVE SHAPE COMPARISON VALIDATION ===\n")
    
    # Test all available categories
    categories = [
        "circle_100",
        "square_50", 
        "rectangle_50",
        # Note: rotated_rectangle_50 and rotated_square_50 were empty/unreadable
    ]
    
    total_correct = 0
    total_images = 0
    category_results = {}
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        
        if not os.path.exists(category_path):
            print(f"⚠️  Category {category} not found, skipping...")
            continue
        
        print(f"Processing {category}:")
        print("-" * 50)
        
        category_correct = 0
        category_total = 0
        errors = []
        
        # Get all PNG files in category
        files = [f for f in os.listdir(category_path) if f.endswith('.png')]
        
        for filename in sorted(files):
            expected_larger = "YES" in filename
            image_path = os.path.join(category_path, filename)
            
            try:
                result = compare_shapes_in_image(image_path)
                
                if not result['success']:
                    errors.append((filename, "No shapes detected", result['error']))
                    category_total += 1
                    continue
                
                # Check if result matches expectation
                shape_1_larger = result['is_shape_1_larger']
                correct = shape_1_larger == expected_larger
                
                if correct:
                    category_correct += 1
                else:
                    shape_1_type = result['shape_1']['type']
                    shape_2_type = result['shape_2']['type']
                    area_ratio = result['area_ratio']
                    errors.append((filename, f"Prediction mismatch", 
                                 f"Expected {expected_larger}, got {shape_1_larger}, types {shape_1_type}/{shape_2_type}, ratio {area_ratio:.3f}"))
                
                category_total += 1
                
            except Exception as e:
                errors.append((filename, "Exception", str(e)))
                category_total += 1
        
        # Calculate category accuracy
        category_accuracy = (category_correct / category_total * 100) if category_total > 0 else 0
        category_results[category] = {
            'correct': category_correct,
            'total': category_total,
            'accuracy': category_accuracy,
            'errors': errors
        }
        
        total_correct += category_correct
        total_images += category_total
        
        print(f"  Results: {category_correct}/{category_total} ({category_accuracy:.1f}% accuracy)")
        
        if errors:
            print(f"  Errors: {len(errors)} files had issues")
            for filename, error_type, details in errors[:3]:  # Show first 3 errors
                print(f"    {filename}: {error_type} - {details}")
            if len(errors) > 3:
                print(f"    ... and {len(errors) - 3} more errors")
        
        print()
    
    # Overall results
    overall_accuracy = (total_correct / total_images * 100) if total_images > 0 else 0
    
    print("=== OVERALL RESULTS ===")
    print(f"Total Images: {total_images}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print()
    
    print("=== CATEGORY BREAKDOWN ===")
    for category, results in category_results.items():
        print(f"{category}: {results['correct']}/{results['total']} ({results['accuracy']:.1f}%)")
    
    print()
    print("=== GROUND TRUTH MEANING ===")
    print("YES labels: Left shape (shape 1) is larger than right shape (shape 2)")
    print("NO labels: Left shape (shape 1) is not larger than right shape (shape 2)")
    print("Shape identification: Based on centroid x-coordinate (left to right)")
    print("Size comparison: Simple area comparison (shape1_area > shape2_area)")
    
    return overall_accuracy

if __name__ == "__main__":
    batch_validate_shape_comparison()
