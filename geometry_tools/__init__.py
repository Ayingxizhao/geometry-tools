"""
Geometry Tools: Computer Vision APIs for Visual Reasoning

A neuro-symbolic framework for solving visual geometry problems,
specifically designed for Müller-Lyer illusion detection and line comparison.

Main Functions:
    - is_line_longer: Compare two lines (returns yes/no answer)
    - measure_line_length: Get precise length of a line
    - compare_all_lines: Auto-detect and compare all lines
    - answer_comparison_question: Natural language interface

Example Usage:
    ```python
    from geometry_tools import is_line_longer, load_image
    
    image = load_image('test.png')
    result = is_line_longer(image, 'line_0', 'line_1')
    
    if result['success']:
        print(f"Answer: {result['answer_text']}")
        print(f"Confidence: {result['confidence']}")
    ```
"""

__version__ = '0.1.0'
__author__ = 'Andrew Y. Zhao'

# Import preprocessing functions
from .preprocessing import (
    load_image,
    preprocess_for_line_detection,
    get_image_info
)

# Import line detection functions
from .line_detection import (
    detect_lines_hough,
    calculate_line_length,
    calculate_line_angle,
    get_line_properties,
    filter_horizontal_lines,
    merge_similar_lines,
    detect_and_filter_lines
)

# Import measurement APIs (main API for LLM)
from .measurements import (
    measure_line_length,
    is_line_longer,
    compare_all_lines,
    answer_comparison_question,
    detect_and_classify_triangles,
    is_valid_equilateral,
    is_valid_right_triangle,
    answer_triangle_question,
    compare_shapes_in_image,
    is_shape_larger,
    answer_shape_comparison_question
)

# Import utility functions
from .utils import (
    draw_lines_on_image,
    highlight_line,
    save_visualization,
    create_comparison_visualization,
    create_synthetic_lines,
    create_muller_lyer_illusion,
    print_detection_summary,
    print_comparison_result
)

# Define public API
__all__ = [
    # Main API functions (for LLM code generation)
    'is_line_longer',
    'measure_line_length',
    'compare_all_lines',
    'answer_comparison_question',
    
    # Triangle detection functions (for LLM code generation)
    'detect_and_classify_triangles',
    'is_valid_equilateral',
    'is_valid_right_triangle',
    'answer_triangle_question',
    
    # Shape comparison functions (for LLM code generation)
    'compare_shapes_in_image',
    'is_shape_larger',
    'answer_shape_comparison_question',
    
    # Preprocessing
    'load_image',
    'preprocess_for_line_detection',
    'get_image_info',
    
    # Line detection
    'detect_lines_hough',
    'detect_and_filter_lines',
    'calculate_line_length',
    'get_line_properties',
    
    # Utilities
    'draw_lines_on_image',
    'save_visualization',
    'create_comparison_visualization',
    'create_synthetic_lines',
    'create_muller_lyer_illusion',
    'print_detection_summary',
    'print_comparison_result',
]