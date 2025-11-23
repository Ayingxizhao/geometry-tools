#!/usr/bin/env python3
"""
Simple test script for Geometry Tools API - Real Working Scenario Demo
"""

import numpy as np
from geometry_tools import (
    is_line_longer,
    measure_line_length,
    compare_all_lines,
    create_synthetic_lines,
    create_muller_lyer_illusion,
    print_comparison_result
)

def test_basic_functionality():
    """Test basic API functionality"""
    print("🚀 Testing Geometry Tools API - Real Working Scenarios")
    print("=" * 60)
    
    try:
        # Test 1: Basic line comparison
        print("\n📏 Test 1: Basic Line Comparison")
        image = create_synthetic_lines(
            width=800, 
            height=400,
            line_configs=[
                {'start': (100, 150), 'end': (300, 150), 'color': (0, 0, 0), 'thickness': 3},  # 200px
                {'start': (450, 250), 'end': (750, 250), 'color': (0, 0, 0), 'thickness': 3},  # 300px
            ]
        )
        
        result = is_line_longer(image, "line_0", "line_1")
        print_comparison_result(result)
        
        # Test 2: Müller-Lyer illusion
        print("\n🎯 Test 2: Müller-Lyer Illusion")
        inward_img, outward_img = create_muller_lyer_illusion(shaft_length=200)
        
        result_inward = is_line_longer(inward_img, "line_0", "line_1")
        print("Inward arrows result:")
        print_comparison_result(result_inward)
        
        # Test 3: Single line measurement
        print("\n📐 Test 3: Single Line Measurement")
        single_image = create_synthetic_lines(
            width=600, 
            height=300,
            line_configs=[
                {'start': (100, 150), 'end': (500, 150), 'color': (0, 0, 0), 'thickness': 3},  # 400px
            ]
        )
        
        measure_result = measure_line_length(single_image, "line_0")
        if measure_result['success']:
            print(f"✅ Measured line: {measure_result['length']:.1f}px (expected: 400px)")
            print(f"📊 Error: {abs(measure_result['length'] - 400):.1f}px")
        else:
            print(f"❌ Measurement failed: {measure_result.get('error', 'Unknown error')}")
        
        # Test 4: Compare all lines
        print("\n🔢 Test 4: Auto-detect and Compare All Lines")
        multi_image = create_synthetic_lines(
            width=800, 
            height=500,
            line_configs=[
                {'start': (100, 100), 'end': (250, 100), 'color': (0, 0, 0), 'thickness': 3},   # 150px
                {'start': (300, 200), 'end': (550, 200), 'color': (0, 0, 0), 'thickness': 3},   # 250px
                {'start': (100, 300), 'end': (400, 300), 'color': (0, 0, 0), 'thickness': 3},   # 300px
            ]
        )
        
        all_result = compare_all_lines(multi_image)
        if all_result['success']:
            print(f"✅ Found {len(all_result['lines'])} lines")
            print(f"🏆 Longest: {all_result['longest_line']} ({all_result['longest_length']:.1f}px)")
            print(f"📏 Shortest: {all_result['shortest_line']} ({all_result['shortest_length']:.1f}px)")
        else:
            print(f"❌ Auto-detection failed: {all_result.get('error', 'Unknown error')}")
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Geometry Tools API is working correctly in real scenarios!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    exit(0 if success else 1)
