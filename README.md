# Geometry Tools: Computer Vision APIs for Visual Reasoning

A neuro-symbolic framework for solving visual geometry problems, specifically designed for Müller-Lyer illusion detection and line comparison.

## 🚀 Quick Start

### Installation

```bash
pip install opencv-python numpy matplotlib
```

### Basic Usage

```python
from geometry_tools import is_line_longer, load_image

image = load_image('test.png')
result = is_line_longer(image, 'line_0', 'line_1')

if result['success']:
    print(f"Answer: {result['answer_text']}")
    print(f"Confidence: {result['confidence']}")
```

## 📦 Main API Functions

### `is_line_longer(image, line_A, line_B, tolerance=0.05)`
Compare two lines and return which is longer.

**Returns:**
```python
{
    'success': True,
    'answer': True,              # Boolean: Is A > B?
    'answer_text': 'yes',        # 'yes', 'no', or 'approximately equal'
    'are_equal': False,          # Within tolerance?
    'line_A_length': 250.5,      # Precise pixel measurements
    'line_B_length': 200.0,
    'difference': 50.5,          # Absolute difference
    'relative_difference': 0.20, # 20%
    'confidence': 'high'         # 'high', 'medium', 'low'
}
```

### `measure_line_length(image, line_id)`
Get precise length of a specific line.

### `compare_all_lines(image)`
Auto-detect and compare all lines in the image.

### `answer_comparison_question(image, question)`
Natural language interface for line comparisons.

## 📁 Project Structure

```
geometry_tools/
├── __init__.py              # Public API exports
├── preprocessing.py         # Image loading & edge detection
├── line_detection.py        # Hough Transform line detection
├── measurements.py          # High-level comparison APIs
├── utils.py                 # Visualization & helpers
└── docs/
    └── PROJECT_SUMMARY.md   # Detailed implementation notes
```
---

**Built with OpenCV and designed for neuro-symbolic visual reasoning.**
