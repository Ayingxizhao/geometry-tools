# Geometry Tools API

A computer vision package for visual geometry reasoning, supporting line comparison, triangle detection, and shape size comparison tasks.

## Quick Start

```python
from geometry_tools import (
    # Line detection (Müller-Lyer illusion)
    is_line_longer,
    compare_all_lines,
    
    # Triangle detection (vision_2d_Check_Triangles)
    detect_and_classify_triangles,
    is_valid_equilateral,
    is_valid_right_triangle,
    
    # Shape comparison (vision_2d_Compare_Size)
    compare_shapes_in_image,
    is_shape_larger,
    answer_shape_comparison_question
)

# Line comparison
result = is_line_longer('image.png', 'line_0', 'line_1')
print(f"Answer: {result['answer_text']}")

# Triangle detection
result = detect_and_classify_triangles('triangle.png')
print(f"Found {result['num_triangles']} triangles")

# Shape comparison
result = compare_shapes_in_image('shapes.png')
print(f"Shape 1 larger: {result['is_shape_1_larger']} (ratio: {result['area_ratio']:.2f})")
```

## Installation

```bash
conda activate contentguard
# Dependencies: opencv-python, numpy, matplotlib
```

## Main Functions

### Line Detection
- `is_line_longer(image, line_A, line_B)` - Compare two lines
- `measure_line_length(image, line_id)` - Measure specific line
- `compare_all_lines(image)` - Auto-detect and compare all lines

### Triangle Detection  
- `detect_and_classify_triangles(image)` - Detect and classify triangles
- `is_valid_equilateral(image, triangle_id)` - Validate equilateral triangles
- `is_valid_right_triangle(image, triangle_id)` - Validate right triangles
- `answer_triangle_question(image, question)` - Natural language interface

### Shape Comparison
- `compare_shapes_in_image(image)` - Detect and compare sizes of shapes (shape 1 vs shape 2)
- `is_shape_larger(image, expected_larger)` - Validate if left shape is larger than right shape
- `answer_shape_comparison_question(image, question)` - Natural language interface for size questions

## Testing

```bash
# Triangle detection validation (99% accuracy on vision_2d_Check_Triangles dataset)
python -m tests.test_batch_validation

# Shape comparison validation (100% accuracy on vision_2d_Compare_Size dataset)
python -m tests.test_shape_batch_validation

# Generate visual debug images
python -m examples.visualize_triangles
python -m examples.visualize_shapes
```

## Documentation

See `docs/` directory for detailed documentation:
- `PROJECT_SUMMARY.md` - Complete implementation overview
- `triangle_detection.md` - Triangle detection API details
- `debugging.md` - Debugging guidelines

## Performance

- **Line detection**: ~99% accuracy, <100ms per image
- **Triangle detection**: 99% accuracy, <200ms per image
- **Supports**: 360p+ resolution images

## File Structure

```
geometry_tools/
├── __init__.py              # Public API
├── preprocessing.py         # Image preprocessing
├── line_detection.py        # Line detection algorithms
├── measurements.py          # Main API functions
├── triangle_detection.py    # Triangle detection algorithms
├── utils.py                 # Visualization utilities
├── docs/                    # Documentation
├── tests/                   # Test suite
└── examples/                # Demo scripts
```
