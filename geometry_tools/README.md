# Geometry Tools API

A computer vision package for visual geometry reasoning, supporting both line comparison and triangle detection tasks.

## Quick Start

```python
from geometry_tools import (
    # Line detection (Müller-Lyer illusion)
    is_line_longer,
    compare_all_lines,
    
    # Triangle detection (vision_2d_Check_Triangles)
    detect_and_classify_triangles,
    is_valid_equilateral,
    is_valid_right_triangle
)

# Line comparison
result = is_line_longer('image.png', 'line_0', 'line_1')
print(f"Answer: {result['answer_text']}")

# Triangle detection
result = detect_and_classify_triangles('triangle.png')
print(f"Found {result['num_triangles']} triangles")
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

## Testing

```bash
# Run batch validation (99% accuracy on vision_2d_Check_Triangles dataset)
python -m tests.test_batch_validation

# Generate visual debug images
python -m examples.visualize_triangles
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
